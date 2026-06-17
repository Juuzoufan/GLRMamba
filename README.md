# GLRMamba

Core model code for **GLRMamba: Normalized Graph-Guided State Space Modeling
for Centralized IoT Roadway Traffic Forecasting**.

This folder intentionally keeps only the main PyTorch model implementation.
It does not include datasets, checkpoints, training logs, ablation scripts, or
the original LibCity experiment framework.

## Files

```text
GLRMamba/
|-- glrmamba.py
|-- requirements.txt
|-- LICENSE
`-- README.md
```

## Scope

This repository is intended for public release of the core model architecture.
It is not a full experimental reproduction package. Raw datasets, checkpoints,
training logs, baseline implementations, and manuscript-specific experiment
scripts are not included.

## Relationship to LibCity

The full experimental workflow of GLRMamba was organized in a LibCity-style
traffic forecasting pipeline. This lightweight repository keeps only the core
PyTorch model for public inspection and reuse.

For the complete urban spatial-temporal prediction framework, dataset format,
and evaluation pipeline, please refer to LibCity:

- Homepage: https://libcity.ai/
- GitHub: https://github.com/LibCity/Bigscity-LibCity
- Documentation: https://bigscity-libcity-docs.readthedocs.io/

## Install

```bash
pip install -r requirements.txt
```

`mamba-ssm` is optional. If it is not installed, `glrmamba.py` uses a small
Conv1d fallback so the model structure can still be imported and smoke-tested.
For reproducing reported results, install `mamba-ssm` in a CUDA-compatible
environment.

## Quick Test

```bash
python glrmamba.py
```

Expected output:

```text
output shape: (2, 12, 170, 1)
```

## Minimal Usage

```python
import torch
from glrmamba import GLRMamba

model = GLRMamba(
    num_nodes=170,
    input_len=12,
    output_len=12,
    input_dim=3,
    output_dim=1,
)

x = torch.randn(8, 12, 170, 3)
time_indices = torch.arange(12)
day_indices = torch.zeros(12, dtype=torch.long)

y = model(x, time_indices=time_indices, day_indices=day_indices)
print(y.shape)  # (8, 12, 170, 1)
```

## Main Components

- RevIN normalization for non-stationary traffic series.
- Mamba temporal block for sequence modeling.
- Graph-guided spatial attention with a learnable bias gate.
- Learnable Fourier temporal encoding for daily and weekly periodicity.
- Parallel horizon predictor with residual-mixture forecasting.

## References

If you use the LibCity framework together with GLRMamba, please cite LibCity:

```bibtex
@inproceedings{libcity,
  author = {Wang, Jingyuan and Jiang, Jiawei and Jiang, Wenjun and Li, Chao and Zhao, Wayne Xin},
  title = {LibCity: An Open Library for Traffic Prediction},
  year = {2021},
  isbn = {9781450386647},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3474717.3483923},
  doi = {10.1145/3474717.3483923},
  booktitle = {Proceedings of the 29th International Conference on Advances in Geographic Information Systems},
  pages = {145--148},
  numpages = {4},
  keywords = {Spatial-temporal System, Reproducibility, Traffic Prediction},
  location = {Beijing, China},
  series = {SIGSPATIAL '21}
}

@article{libcitylong,
  title = {LibCity: A Unified Library Towards Efficient and Comprehensive Urban Spatial-Temporal Prediction},
  author = {Jiang, Jiawei and Han, Chengkai and Jiang, Wenjun and Zhao, Wayne Xin and Wang, Jingyuan},
  journal = {arXiv preprint arXiv:2304.14343},
  year = {2023}
}
```
