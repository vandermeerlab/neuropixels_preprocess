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

> **If the recording contains one or more quadprobes, always set `TO_PROBE` to a quadprobe.**
> A probe is trivially aligned to itself, so making the quadprobe the reference means its shanks
> never need per-shank spike alignment at all — only a second quadprobe in the same recording
> would.

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
CATGT_OUTPUT_DIR    = E:\data\catgt_M511-2024-11-14_g0
OUTPUT_DIR          = E:\data\preprocessed
SESSION_ID          = M511-2024-11-14
FAST_STORAGE_DIR    = F:\KS4_outputs
PROBE_INDICES       = 0,1
PROCESS_BY_SHANK_FOR =
MAX_BAD_CHANNELS    = 25
```

`PROCESS_BY_SHANK_FOR` lists which probes (from `PROBE_INDICES`) should be split by shank; leave it empty for standard whole-probe processing. See Workflow 2 below for the per-shank case, and Workflow 3 for sessions that mix both within the same run.

```
python extract_lfp_and_run_ks4.py --file_params example_lfp_ks4_params.txt
```

Outputs per probe:
- `imec{p}_clean_lfp.mat` — LFP traces and depth info
- `imec{p}_lfp_times.npy` — LFP time vector (used by TPrime)
- `{SESSION_ID}_imec{p}/` — Kilosort4 output folder
- `imec{p}_bad_channels.json` — detected (or manually supplied) bad channel IDs, written to `CATGT_OUTPUT_DIR`

#### Manual bad channel curation

By default, bad channels are auto-detected per probe (or per shank — see Workflow 2/3) with `si.detect_bad_channels`, and written to `imec{p}_bad_channels.json` (or `imec{p}_shank{s}_bad_channels.json`) in `CATGT_OUTPUT_DIR`. If a probe/shank exceeds `MAX_BAD_CHANNELS`, it's flagged for manual review instead of being processed automatically.

To override auto-detection, set `OVERRIDE_BAD_CHANNELS = True` and point at bad channel JSON files with one of:

- **A dedicated file per probe/shank** — the same flat format `{"bad_channel_ids": [...]}` this script itself writes, so you can hand-edit the auto-detected file and feed it back in:
  ```
  OVERRIDE_BAD_CHANNELS = True
  MANUAL_BAD_CHANNELS_FILE_IMEC0        = E:\data\catgt_..._g0\imec0_bad_channels.json
  MANUAL_BAD_CHANNELS_FILE_IMEC0_SHANK0 = E:\data\catgt_..._g0\imec0_shank0_bad_channels.json
  ```
  Add one `MANUAL_BAD_CHANNELS_FILE_IMEC{p}` (whole-probe) or `MANUAL_BAD_CHANNELS_FILE_IMEC{p}_SHANK{s}` (per-shank) line per probe/shank you want to override — only the ones you list are overridden, so this mixes cleanly with auto-detection for the rest.
- **One combined file for the whole session**, with nested per-probe/shank keys:
  ```
  MANUAL_BAD_CHANNELS_FILE = path/to/manual_bad_channels.json
  ```
  where the file looks like `{"imec0_shank0": {"bad_channel_ids": [...]}, "imec1": {"bad_channel_ids": [...]}}`. Per-probe/shank overrides above take precedence over this if both are set.

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
KS_OUTPUT_FOLDERS = {"imec1": "F:\\KS4_outputs\\M511-2024-11-14_imec1\\sorter_output"}
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
SORTER_OUTPUT_FOLDERS = {"imec0": "F:\\KS4_outputs\\M511-2024-11-14_imec0\\sorter_output", "imec1": "F:\\KS4_outputs\\M511-2024-11-14_imec1\\sorter_output"}
PREPROCESS_FOLDERS    = {"imec0": "F:\\KS4_outputs\\M511-2024-11-14_imec0_si_preprocess", "imec1": "F:\\KS4_outputs\\M511-2024-11-14_imec1_si_preprocess"}
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
CATGT_OUTPUT_DIR    = E:\data\catgt_MM019-2026-08-15_g0
OUTPUT_DIR          = E:\data\preprocessed
SESSION_ID          = MM019-2026-08-15
FAST_STORAGE_DIR    = F:\KS4_outputs
PROBE_INDICES       = 0
PROCESS_BY_SHANK_FOR = 0
MAX_BAD_CHANNELS    = 100
```

```
python extract_lfp_and_run_ks4.py --file_params example_lfp_ks4_params.txt
```

Outputs per shank:
- `imec{p}_shank{s}_clean_lfp.mat`
- `{SESSION_ID}_imec{p}_shank{s}/` — Kilosort4 output folder

Plus one file for the probe as a whole:
- `imec{p}_lfp_times.npy` — the shanks are clock-locked, so they share a single LFP time vector
  (verified identical across all four shanks on real Quadbase data). Downstream MATLAB reads
  `imec{p}_adj_lfp_times.npy` and needs no shank-aware changes.

### Step 4: TPrime alignment (per-shank)

With a single quadprobe as `TO_PROBE` there is nothing to align per-shank — it is the reference
clock — so `PROCESS_BY_SHANK_FOR` only matters for a *second* quadprobe being aligned to it:

```
TO_PROBE             = 0
PROCESS_BY_SHANK_FOR = 1
SHANK_INDICES        = 0,1,2,3
KS_OUTPUT_FOLDERS    = {"imec1_shank0": "F:\\KS4_outputs\\MM019-2026-08-15_imec1_shank0\\sorter_output", "imec1_shank1": "F:\\KS4_outputs\\MM019-2026-08-15_imec1_shank1\\sorter_output", "imec1_shank2": "F:\\KS4_outputs\\MM019-2026-08-15_imec1_shank2\\sorter_output", "imec1_shank3": "F:\\KS4_outputs\\MM019-2026-08-15_imec1_shank3\\sorter_output"}
```

```
python run_tprime.py --file_params example_tprime_params.txt
```

### Step 5: Extract curated units (per-shank)

```
PROCESS_BY_SHANK_FOR  = 0
SHANK_INDICES         = 0,1,2,3
SORTER_OUTPUT_FOLDERS = {"imec0_shank0": "F:\\KS4_outputs\\MM019-2026-08-15_imec0_shank0\\sorter_output", "imec0_shank1": "...", "imec0_shank2": "...", "imec0_shank3": "..."}
PREPROCESS_FOLDERS    = {"imec0_shank0": "F:\\KS4_outputs\\MM019-2026-08-15_imec0_shank0_si_preprocess", "imec0_shank1": "...", "imec0_shank2": "...", "imec0_shank3": "..."}
```

Units from all shanks are merged into a single output file per probe (`good_units_imec{p}.mat`), so downstream loading is unchanged.

```
python extract_units_after_curation.py --file_params example_extract_postcuration_params.txt
```

---

## Workflow 3 — Mixed multi-probe (some probes by shank, some whole)

For a session with more than one probe where they don't all need the same treatment — e.g. probe 0 is a Quadbase probe sorted per-shank and probe 1 is a standard NPX2 probe sorted whole.

### Step 2: LFP extraction and spike sorting (mixed)

`PROCESS_BY_SHANK_FOR` takes a list, so only the probes that need shank-splitting go in it — everything else in `PROBE_INDICES` is processed whole in the same run:

```
CATGT_OUTPUT_DIR     = E:\data\catgt_M700-2026-08-20_g0
OUTPUT_DIR           = E:\data\preprocessed
SESSION_ID           = M700-2026-08-20
FAST_STORAGE_DIR     = F:\KS4_outputs
PROBE_INDICES        = 0,1
PROCESS_BY_SHANK_FOR = 0
MAX_BAD_CHANNELS     = 100
```

Manual bad channel overrides mix the same way — per-shank keys for probe 0, a plain per-probe key for probe 1:

```
OVERRIDE_BAD_CHANNELS = True
MANUAL_BAD_CHANNELS_FILE_IMEC0_SHANK0 = E:\data\catgt_..._g0\imec0_shank0_bad_channels.json
MANUAL_BAD_CHANNELS_FILE_IMEC0_SHANK1 = E:\data\catgt_..._g0\imec0_shank1_bad_channels.json
MANUAL_BAD_CHANNELS_FILE_IMEC0_SHANK2 = E:\data\catgt_..._g0\imec0_shank2_bad_channels.json
MANUAL_BAD_CHANNELS_FILE_IMEC0_SHANK3 = E:\data\catgt_..._g0\imec0_shank3_bad_channels.json
MANUAL_BAD_CHANNELS_FILE_IMEC1        = E:\data\catgt_..._g0\imec1_bad_channels.json
```

```
python extract_lfp_and_run_ks4.py --file_params example_lfp_ks4_params.txt
```

This gives you `imec0_shank{0-3}_clean_lfp.mat` + `{SESSION_ID}_imec0_shank{0-3}/` for probe 0, and `imec1_clean_lfp.mat` + `{SESSION_ID}_imec1/` for probe 1, in the same run.

### Step 4: TPrime alignment (mixed)

**Set `TO_PROBE` to the quadprobe.** Probe 0 is then the reference clock and needs no alignment of
its own, so with a single quadprobe in the recording `PROCESS_BY_SHANK_FOR` can stay empty here —
the only probes being aligned (1 and 2) are whole probes:

```
PROBE_INDICES        = 0,1,2
TO_PROBE             = 0
PROCESS_BY_SHANK_FOR =
KS_OUTPUT_FOLDERS    = {"imec1": "F:\\KS4_outputs\\M700-2026-08-20_imec1\\sorter_output", "imec2": "F:\\KS4_outputs\\M700-2026-08-20_imec2\\sorter_output"}
```

If a *second* quadprobe (say probe 2) is also present, list it and mix both key styles in one dict:

```
PROCESS_BY_SHANK_FOR = 2
SHANK_INDICES        = 0,1,2,3
KS_OUTPUT_FOLDERS    = {"imec1": "...\\imec1\\sorter_output", "imec2_shank0": "...\\imec2_shank0\\sorter_output", "imec2_shank1": "...", "imec2_shank2": "...", "imec2_shank3": "..."}
```

Each probe is validated against its own mode, so the two styles coexist. LFP alignment is always
per-probe (`imec{p}_lfp_times.npy` → `imec{p}_adj_lfp_times.npy`) regardless of shank mode.

### Step 5: Extract curated units (mixed)

Unlike Step 4 there is no reference probe here — every probe in `PROBE_INDICES` is processed — so
the by-shank probe must always be listed:

```
PROBE_INDICES         = 0,1
PROCESS_BY_SHANK_FOR  = 0
SHANK_INDICES         = 0,1,2,3
SORTER_OUTPUT_FOLDERS = {"imec0_shank0": "...", "imec0_shank1": "...", "imec0_shank2": "...", "imec0_shank3": "...", "imec1": "F:\\KS4_outputs\\M700-2026-08-20_imec1\\sorter_output"}
PREPROCESS_FOLDERS    = {"imec0_shank0": "...", "imec0_shank1": "...", "imec0_shank2": "...", "imec0_shank3": "...", "imec1": "F:\\KS4_outputs\\M700-2026-08-20_imec1_si_preprocess"}
```

Probe 0's four shanks are merged into one `good_units_imec0.mat`; probe 1 is extracted whole into
`good_units_imec1.mat`. Downstream loading is identical for both.

> If two by-shank probes have *different* shank sets, override per probe with
> `SHANK_INDICES_IMEC{p} = 0,1` — it takes precedence over the global `SHANK_INDICES` for that probe.
> This applies to both Step 4 and Step 5.
