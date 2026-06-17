# Data Preparation

This project expects the PEMSD datasets in LibCity traffic-state format.
Dataset files are not stored in the repository because of size and distribution
constraints.

## Required Directory Layout

Place files under `raw_data/<DATASET>/`:

```text
raw_data/PEMSD8/PEMSD8.geo
raw_data/PEMSD8/PEMSD8.rel
raw_data/PEMSD8/PEMSD8.dyna
raw_data/PEMSD8/config.json
```

Repeat the same pattern for `PEMSD3`, `PEMSD4`, and `PEMSD7`.

## Required File Types

- `.geo`: sensor metadata. Must contain a `geo_id` column.
- `.rel`: graph relation file. Must contain source and destination sensor ids
  and a cost/link column compatible with `config.json`.
- `.dyna`: time-indexed traffic observations. The default configs expect
  traffic-flow, occupancy, and speed columns for PEMSD3/4/8.
- `config.json`: LibCity dataset metadata. It should define file names,
  columns, time interval, relation settings, and dataset type information.

## Feature Layout

For PEMSD3, PEMSD4, and PEMSD8, the default model config uses:

```text
traffic_flow, traffic_occupancy, traffic_speed
```

The pipeline can append time-of-day and day-of-week features when
`load_external`, `add_time_in_day`, and `add_day_in_week` are enabled.

For PEMSD7, this repository includes `preprocess_pemsd7.py` for cases where
only traffic-flow is available. The script creates a cache with the feature
layout:

```text
traffic_flow, occupancy_placeholder, speed_placeholder, time_in_day, day_of_week_one_hot
```

Run it from the repository root after placing `raw_data/PEMSD7/`:

```bash
python preprocess_pemsd7.py
```

The generated cache is written to:

```text
libcity/cache/dataset_cache/
```

## Cache Behavior

Processed dataset caches are created under `libcity/cache/dataset_cache/`.
Delete the corresponding cache file if you change a dataset config and want the
pipeline to regenerate train/validation/test windows.

