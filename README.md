# GLRMamba Reproducibility Package

This repository contains the executable implementation used for
**GLRMamba: Normalized Graph-Guided State Space Modeling for Centralized IoT
Roadway Traffic Forecasting**.

The code is organized as a lightweight LibCity-style traffic forecasting
pipeline. The model class is named `MambaFormer` in the pipeline for historical
compatibility, but it implements the GLRMamba architecture described in the
manuscript. A standalone PyTorch version is also provided in `glrmamba.py`.

## Files

```text
GLRMamba/
|-- glrmamba.py
|-- run_model.py
|-- mambaformer_pemsd{3,4,7,8}_config.json
|-- libcity/
|   |-- data/
|   |-- evaluator/
|   |-- executor/
|   |-- model/traffic_flow_prediction/MambaFormer.py
|   `-- pipeline/
|-- raw_data/README.md
|-- DATA_PREPARATION.md
|-- REPRODUCIBILITY_STATUS.md
|-- requirements.txt
|-- LICENSE
`-- README.md
```

## Scope

This repository provides the proposed GLRMamba model, the LibCity-style
training/evaluation pipeline, dataset configuration files, and data preparation
instructions for the main experiments.

Raw PEMSD datasets, large generated caches, full checkpoints, and third-party
baseline implementations are not included. Ablation, hyperparameter-sensitivity,
and visualization scripts are kept out of this public repository to keep the
release focused on the main experimental pipeline.

## Relationship to LibCity

The experimental workflow of GLRMamba follows a LibCity-style traffic
forecasting pipeline. The included `libcity/` folder is a compact, adapted
subset used to run the GLRMamba experiments.

For the complete urban spatial-temporal prediction framework, dataset format,
and evaluation pipeline, please refer to LibCity:

- Homepage: https://libcity.ai/
- GitHub: https://github.com/LibCity/Bigscity-LibCity
- Documentation: https://bigscity-libcity-docs.readthedocs.io/

## Install

The reported experiments were run with Python 3.8, PyTorch 1.12.1, CUDA 11.3,
and an NVIDIA GPU.

```bash
conda create -n glrmamba python=3.8 -y
conda activate glrmamba
pip install -r requirements.txt
```

For the reported GLRMamba setting, install `mamba-ssm` separately in a
CUDA-enabled Linux environment compatible with your PyTorch/CUDA version. If
`mamba-ssm` is unavailable, the code falls back to a Conv1d temporal block so
that the pipeline can still be inspected and smoke-tested; this fallback is
not the reported experimental setting.

## Data

Raw PEMSD datasets are not committed to this repository. Place each dataset
under `raw_data/<DATASET>/` in LibCity format:

```text
raw_data/
  PEMSD3/
    PEMSD3.geo
    PEMSD3.rel
    PEMSD3.dyna
    config.json
  PEMSD4/
    PEMSD4.geo
    PEMSD4.rel
    PEMSD4.dyna
    config.json
  PEMSD7/
    PEMSD7.geo
    PEMSD7.rel
    PEMSD7.dyna
    config.json
  PEMSD8/
    PEMSD8.geo
    PEMSD8.rel
    PEMSD8.dyna
    config.json
```

See `DATA_PREPARATION.md` and `raw_data/README.md` for details.

## Run Main Experiments

```bash
python run_model.py --config_file mambaformer_pemsd3_config
python run_model.py --config_file mambaformer_pemsd4_config
python run_model.py --config_file mambaformer_pemsd7_config
python run_model.py --config_file mambaformer_pemsd8_config
```

Generated checkpoints and evaluation files are written to:

```text
libcity/cache/<exp_id>/model_cache/
libcity/cache/<exp_id>/evaluate_cache/
libcity/log/
```

## Standalone Model Quick Test

```bash
python glrmamba.py
```

Expected output:

```text
output shape: (2, 12, 170, 1)
```

## Minimal Standalone Usage

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
