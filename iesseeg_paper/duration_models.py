"""Ridge probe matching the original duration study's CUDA linear maps."""
import numpy as np
import torch


class MatchedRidge:
    """Dimension-normalized ridge with an unpenalized intercept.

    Preserve the original map calculation, including centering after the solve,
    to avoid rank/validation-tie differences from another solver's recentering.
    Inputs have already been transformed by training-only preprocessing.
    """
    def __init__(self, alpha, device):
        self.alpha = alpha
        self.device = str(device)

    def fit(self, x, y):
        self.train = np.asarray(x, dtype=np.float64)
        self.targets = 2 * np.asarray(y) - 1
        return self

    def decision_function(self, x):
        a = torch.as_tensor(self.train, dtype=torch.float64, device=self.device)
        b = torch.as_tensor(x, dtype=torch.float64, device=self.device)
        d = max(1, a.shape[1])
        kernel = a @ a.T / d
        cross = b @ a.T / d
        identity = torch.eye(len(a), dtype=torch.float64, device=self.device)
        h = torch.linalg.solve(kernel + len(a) * self.alpha * identity, cross.T).T
        h = h - h.mean(dim=1, keepdim=True) + 1 / len(a)
        return np.einsum('vt,t->v', h.cpu().numpy(), self.targets)
