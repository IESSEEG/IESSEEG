"""Independent neural-head refits batched over label assignments.

All assignments share a prespecified model initialization, recording order,
and dropout stream. Parameters, optimizer moments, and PSMIL banks remain
separate. This conditions the randomization test on algorithmic randomness.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "legacy/analysis"))
import torch
from torch.func import functional_call
from run_clip_response_screen import RecordingHead


class CompatibleHead(RecordingHead):
    @staticmethod
    def _mask_deciles(h, ratio=0.5):
        if ratio <= 0 or len(h) < 10:
            return h
        n = int(ratio * 10)
        selected = torch.randperm(10, device=h.device)[:n]
        value = torch.randn(n, device=h.device, dtype=h.dtype)
        position = torch.arange(len(h), device=h.device) // (len(h) // 10)
        match = position[:, None] == selected[None, :]
        replacement = (match * value).sum(1)
        return torch.where(match.any(1)[:, None], replacement[:, None], h)

    def _psmil(self, h, label, update):
        with torch.no_grad():
            initial = h.detach().mean(0)[:, None].expand(-1, 2)
            self.ps_prototypes.copy_(
                torch.where(self.ps_initialized, self.ps_prototypes, initial)
            )
            self.ps_initialized.fill_(True)
        bank = self.ps_prototypes.detach().clone()
        probability = torch.softmax(h @ bank, dim=1)
        weight = torch.softmax(self.ps_attention(probability).squeeze(-1), dim=0)
        logits = (weight[:, None] * h).sum(0, keepdim=True) @ self.ps_classifier
        if update:
            with torch.no_grad():
                which = torch.stack([1 - label, label]).to(h.dtype)
                selected = torch.argmax((probability * which).sum(1))
                critical = (
                    h.gather(0, selected.reshape(1, 1).expand(1, h.shape[1]))
                    .squeeze(0)
                    .detach()
                )
                old = (self.ps_prototypes * which).sum(1)
                updated = torch.nn.functional.normalize(
                    0.99 * old + 0.01 * critical, dim=0
                )
                self.ps_prototypes.copy_(
                    self.ps_prototypes * (1 - which) + updated[:, None] * which
                )
        return (logits[0, 1] - logits[0, 0]).reshape(())


class BatchedHeads:
    def __init__(self, dimension, method, n, device, seed, dropout=0.2):
        torch.manual_seed(seed)
        self.template = CompatibleHead(dimension, method, dropout=dropout).to(device)
        self.method = method
        self.n = n
        self.params = {
            k: torch.nn.Parameter(
                v.detach().unsqueeze(0).expand(n, *v.shape).clone(),
                requires_grad=v.requires_grad,
            )
            for k, v in self.template.named_parameters()
        }
        self.buffers = {
            k: v.detach().unsqueeze(0).expand(n, *v.shape).clone()
            for k, v in self.template.named_buffers()
        }
        self.optimizer = torch.optim.AdamW(
            self.params.values(), lr=3e-4, weight_decay=1e-3
        )

        # Capture the module, not this wrapper, so completed fits release their
        # parameter/optimizer tensors immediately instead of forming a cycle.
        template = self.template

        def single(params, buffers, x, epoch, label, update):
            return functional_call(
                template,
                (params, buffers),
                (x,),
                {"epoch": epoch, "label": label, "update_psmil": update},
            )

        self.batched = torch.vmap(
            single, in_dims=(0, 0, None, None, 0, None), randomness="same"
        )

    def forward(self, x, labels, epoch=0, training=False):
        self.template.train(training)
        return self.batched(
            self.params,
            self.buffers,
            x,
            epoch,
            labels,
            training and self.method == "psmil",
        )

    def step(self):
        # Clip each independently fitted model, never the combined family norm.
        terms = [
            p.grad.square().reshape(self.n, -1).sum(1)
            for p in self.params.values()
            if p.grad is not None
        ]
        norm = torch.stack(terms).sum(0).sqrt()
        factor = (5.0 / (norm + 1e-6)).clamp(max=1.0)
        for p in self.params.values():
            if p.grad is not None:
                p.grad.mul_(factor.reshape(self.n, *([1] * (p.ndim - 1))))
        self.optimizer.step()

    def compact(self, keep):
        """Drop completed assignments, retaining every active optimizer moment."""
        old_params = self.params
        old_opt = self.optimizer
        self.params = {
            k: torch.nn.Parameter(
                v.detach()[keep].clone(), requires_grad=v.requires_grad
            )
            for k, v in old_params.items()
        }
        self.buffers = {k: v.detach()[keep].clone() for k, v in self.buffers.items()}
        self.n = int(keep.sum())
        self.optimizer = torch.optim.AdamW(
            self.params.values(), lr=3e-4, weight_decay=1e-3
        )
        for k, old in old_params.items():
            if old not in old_opt.state:
                continue
            self.optimizer.state[self.params[k]] = {
                name: (
                    value[keep].clone()
                    if isinstance(value, torch.Tensor) and value.ndim > 0
                    else value
                )
                for name, value in old_opt.state[old].items()
            }

    def train_epoch(self, windows, labels, order, epoch, pos_weight, batch_size=8):
        for start in range(0, len(order), batch_size):
            self.optimizer.zero_grad(set_to_none=True)
            batch = order[start : start + batch_size]
            losses = []
            for i in batch:
                logits = self.forward(windows[int(i)], labels[:, int(i)], epoch, True)
                losses.append(
                    torch.nn.functional.binary_cross_entropy_with_logits(
                        logits,
                        labels[:, int(i)].float(),
                        pos_weight=pos_weight,
                        reduction="none",
                    )
                )
            torch.stack(losses).mean(0).sum().backward()
            self.step()

    def predict(self, windows):
        dummy = torch.zeros(self.n, device=windows[0].device, dtype=torch.long)
        with torch.no_grad():
            return (
                torch.stack(
                    [torch.sigmoid(self.forward(x, dummy)) for x in windows], dim=1
                )
                .cpu()
                .numpy()
            )
