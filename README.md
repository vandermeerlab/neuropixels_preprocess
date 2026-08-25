# neuropixels_preprocess

Scripts and documentation for preprocessing neural and auxiliary data collected with Neuropixels probes.

## Environment setup

See [EnvironmentSetup.md](EnvironmentSetup.md) for exact package versions and install instructions.

## Pipeline overview

The pipeline runs in four sequential steps:

| Step | Script | Purpose |
|---|---|---|
| 1 | `run_catgt.py` | Timestamp-correct raw data, extract NIDQ sync/events |
| 2 | `extract_lfp_and_run_ks4.py` | Bad channel detection, LFP extraction, Kilosort4 spike sorting |
| 3 | `run_tprime.py` | Align spike/LFP times across probes and to NIDQ clock |
| 4 | `extract_units_after_curation.py` | Extract curated units (after Phy) into `.mat` files |

Each script takes a params file: `python script.py --file_params path/to/params.txt`

For batch processing across sessions: `python script.py --batch_params path/to/session_list.txt`

---

## Workflow 1 — Standard NPX2 (per-probe)

For probes with a single shank or where all channels are sorted together (e.g. NPX 2.0 single).

### Step 1: CatGT

Edit `example_catgt_params.txt`:

```
SOURCE_DIR = E:\data\session_g0
DEST_DIR   = E:\data
SESSION_NAME = M511-2024-11-14
CATGT_PATH   = D:\CatGT-win
PROBE_INDICES = 0,1
ANALOG_CHANNELS = {"XA1": [1,6], "XA2": [1,3]}
DIGITAL_CHANNELS = XD0,XD1,XD2,XD3
```

```
python run_catgt.py --file_params example_catgt_params.txt
```

> To process only NIDQ channels (no AP probe data), set `PROBE_INDICES =` (leave blank or omit).

### Step 2: LFP extraction and spike sorting

Edit `example_lfp_ks4_params.txt`:

```
CATGT_OUTPUT_DIR = E:\data\catgt_M511-2024-11-14_g0
OUTPUT_DIR       = E:\data\preprocessed
SESSION_ID       = M511-2024-11-14
FAST_STORAGE_DIR = F:\KS4_outputs
PROBE_INDICES    = 0,1
PROCESS_BY_SHANK = False
MAX_BAD_CHANNELS = 25
```

```
python extract_lfp_and_run_ks4.py --file_params example_lfp_ks4_params.txt
```

Outputs per probe:
- `imec{p}_clean_lfp.mat` — LFP traces and depth info
- `imec{p}_lfp_times.npy` — LFP time vector (used by TPrime)
- `{SESSION_ID}_imec{p}/` — Kilosort4 output folder

### Step 3: Curate in Phy

Open each Kilosort4 output in Phy and label units as `good` / `mua`.

### Step 4: TPrime alignment

Edit `example_tprime_params.txt`:

```
CATGT_OUTPUT_DIR  = E:\data\catgt_M511-2024-11-14_g0
OUTPUT_DIR        = E:\data\preprocessed
SESSION_ID        = M511-2024-11-14
TPRIME_PATH       = D:\TPrime-win
PROBE_INDICES     = 0,1
TO_PROBE          = 0
PROCESSED_BY_SHANK = False
KS_OUTPUT_FOLDERS = {"1": "F:\\KS4_outputs\\M511-2024-11-14_imec1\\sorter_output"}
```

```
python run_tprime.py --file_params example_tprime_params.txt
```

### Step 5: Extract curated units

Edit `example_extract_postcuration_params.txt`:

```
CATGT_OUTPUT_DIR      = E:\data\catgt_M511-2024-11-14_g0
OUTPUT_DIR            = E:\data\preprocessed
SESSION_ID            = M511-2024-11-14
PROBE_INDICES         = 0,1
PROCESSED_BY_SHANK    = False
SORTER_OUTPUT_FOLDERS = {"0": "F:\\KS4_outputs\\M511-2024-11-14_imec0\\sorter_output", "1": "F:\\KS4_outputs\\M511-2024-11-14_imec1\\sorter_output"}
PREPROCESS_FOLDERS    = {"0": "F:\\KS4_outputs\\M511-2024-11-14_imec0_si_preprocess", "1": "F:\\KS4_outputs\\M511-2024-11-14_imec1_si_preprocess"}
```

```
python extract_units_after_curation.py --file_params example_extract_postcuration_params.txt
```

---

## Workflow 2 — Quadbase (per-shank)

For the Neuropixels Quadbase probe (4 shanks × 384 channels). Each shank is sorted independently.

Steps 1 and 3 (CatGT, Phy curation) are the same as Workflow 1. The differences are in the params for Steps 2, 4, and 5.

### Step 2: LFP extraction and spike sorting (per-shank)

```
CATGT_OUTPUT_DIR = E:\data\catgt_MM019-2026-08-15_g0
OUTPUT_DIR       = E:\data\preprocessed
SESSION_ID       = MM019-2026-08-15
FAST_STORAGE_DIR = F:\KS4_outputs
PROBE_INDICES    = 0
PROCESS_BY_SHANK = True
MAX_BAD_CHANNELS = 100
```

```
python extract_lfp_and_run_ks4.py --file_params example_lfp_ks4_params.txt
```

Outputs per shank:
- `imec{p}_shank{s}_clean_lfp.mat`
- `imec{p}_shank{s}_lfp_times.npy`
- `{SESSION_ID}_imec{p}_shank{s}/` — Kilosort4 output folder

### Step 4: TPrime alignment (per-shank)

```
PROCESSED_BY_SHANK = True
SHANK_INDICES      = 0,1,2,3
KS_OUTPUT_FOLDERS  = {"0_shank0": "F:\\KS4_outputs\\MM019-2026-08-15_imec0_shank0\\sorter_output", "0_shank1": "F:\\KS4_outputs\\MM019-2026-08-15_imec0_shank1\\sorter_output", "0_shank2": "F:\\KS4_outputs\\MM019-2026-08-15_imec0_shank2\\sorter_output", "0_shank3": "F:\\KS4_outputs\\MM019-2026-08-15_imec0_shank3\\sorter_output"}
```

```
python run_tprime.py --file_params example_tprime_params.txt
```

### Step 5: Extract curated units (per-shank)

```
PROCESSED_BY_SHANK    = True
SHANK_INDICES         = 0,1,2,3
SORTER_OUTPUT_FOLDERS = {"0_shank0": "F:\\KS4_outputs\\MM019-2026-08-15_imec0_shank0\\sorter_output", "0_shank1": "...", "0_shank2": "...", "0_shank3": "..."}
PREPROCESS_FOLDERS    = {"0_shank0": "F:\\KS4_outputs\\MM019-2026-08-15_imec0_shank0_si_preprocess", "0_shank1": "...", "0_shank2": "...", "0_shank3": "..."}
```

Units from all shanks are merged into a single output file per probe (`good_units_imec{p}.mat`), so downstream loading is unchanged.

```
python extract_units_after_curation.py --file_params example_extract_postcuration_params.txt
```
