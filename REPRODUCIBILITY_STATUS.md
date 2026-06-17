# Reproducibility Status

Last updated: 2026-06-17

## Included

- GLRMamba implementation in `libcity/model/traffic_flow_prediction/MambaFormer.py`.
- Main training/evaluation entry point: `run_model.py`.
- Dataset configs for PEMSD3, PEMSD4, PEMSD7, and PEMSD8.
- Requirements file and data preparation instructions.

## Not Included

- Raw PEMSD dataset files.
- Full model checkpoints for all reported experiments.
- Third-party baseline implementations.
- Ablation, hyperparameter-sensitivity, and visualization scripts.

## Smoke Checks

The following local check has been run:

```bash
python -m compileall -q run_model.py glrmamba.py libcity
```

Result: passed.

Full training was not run in this workspace because raw PEMSD data are not
present under `raw_data/`.
