# Third-party code and model weights

The project-level MIT license covers the IESSEEG project code copied from its existing release and the additions in this package. It does not replace third-party licenses.

| Component | Origin | Notice in this repository |
|---|---|---|
| BIOT implementation | [ycq091044/BIOT](https://github.com/ycq091044/BIOT) | `baselines_reference/baselines/biot/LICENSE.upstream` (MIT, Chaoqi Yang) |
| LaBraM implementation | [935963004/LaBraM](https://github.com/935963004/LaBraM) | `baselines_reference/baselines/labram/LICENSE.upstream` (MIT, Weibang Jiang); source attribution headers retained |
| CBraMod implementation | [wjq-learning/CBraMod](https://github.com/wjq-learning/CBraMod) | `baselines_reference/baselines/cbramod/LICENSE.upstream` (MIT, Jiquan Wang); [upstream license](https://github.com/wjq-learning/CBraMod/blob/main/LICENSE) checked 20 September 2026 |
| LUNA implementation | [pulp-bio/BioFoundation](https://github.com/pulp-bio/BioFoundation) | `baselines_reference/baselines/luna/LICENSE.upstream` (Apache-2.0); ETH Zurich notices retained |
| EEGPT | Braindecode implementation and externally obtained pretrained weights | External dependency; see docs/MODELS.md |
| REVE | External Hugging Face model implementation and weights | External dependency; wrapper only |
| CodeBrain | External upstream source and weights | Wrapper only; see docs/MODELS.md |
| CSBrain | External upstream source and weights | Wrapper only; no upstream source or weights redistributed |

Adaptations in this package redirect inputs and outputs, retain the original split procedures, and expose the benchmark-specific montage, preprocessing, and training settings. Existing upstream attribution comments are preserved. No pretrained or fine-tuned model weights are included in the publication folder. Dependencies and separately downloaded weights retain their own terms.

EEG and linked clinical annotations require their own data license and access policy. The project's software license does not grant rights to redistribute or commercially use those data.

The REVE StableAdamW optimizer under `third_party/reve_optimizer/` is preserved
from the official REVE implementation, including its upstream license and
embedded attribution. Project-owned copyright attribution is anonymized during
review; it does not replace any third-party copyright notice.
