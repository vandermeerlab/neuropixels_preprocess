#!/usr/bin/env python3
"""
Neuropixels LFP Extraction and Kilosort4 Processing

Supports both per-probe (standard NPX2) and per-shank (e.g. Quadbase: 4 shanks x 384 ch)
processing, controlled per-probe by the PROCESS_BY_SHANK_FOR parameter.

Usage:
    python extract_lfp_and_run_ks4.py --file_params path/to/params.txt
    python extract_lfp_and_run_ks4.py --batch_params path/to/batch_list.txt
"""

import os
import re
import sys
import json
import logging
import numpy as np
import scipy.io as scio
import argparse
import shutil
from datetime import datetime
import spikeinterface.full as si

# Default parameters
DEFAULT_PARAMS = {
    'CATGT_OUTPUT_DIR': None,   # Required
    'OUTPUT_DIR': None,         # Required
    'SESSION_ID': None,         # Required
    'FAST_STORAGE_DIR': None,   # Required
    'PROBE_INDICES': [0],
    'EXTRACT_LFP_FOR': None,
    'RUN_KILOSORT_FOR': None,
    'PROCESS_BY_SHANK_FOR': [],  # probe indices to split by shank (e.g. Quadbase); other probes processed whole
    'OVERRIDE_BAD_CHANNELS': False,
    'MANUAL_BAD_CHANNELS_FILE': None,   # single combined file (nested imecN / imecN_shankN keys)
    'MANUAL_BAD_CHANNELS_FILES': {},    # per-probe/shank overrides, keyed by e.g. 'imec0_shank0'; see MANUAL_BAD_CHANNELS_FILE_IMEC*_SHANK* params
    'MAX_BAD_CHANNELS': 25,     # per shank/probe; set ~100 for Quadbase (384 ch/shank)
    'OVERRIDE_LFP_CHANNELS': False,
    'MANUAL_LFP_CHANNELS_FILE': None,
    'LFP_SAMPLE_RATE': 2500,
    'LFP_HIGH_PASS': 1,
    'LFP_LOW_PASS': 400,
    'LFP_SPACING': 100,
    'INTER_SHANK_SPACE': 250,
    'AP_HIGH_PASS': 400.0,
    'KS_PARAMS': {
        'delete_recording_dat': True,
        'use_binary_file': True
    },
    'LFP_NUM_JOBS': 8,
    'SI_NUM_JOBS': 10,
    'CHUNK_DURATION': '1s',
    'LFP_CHUNK_DURATION': '30s',
}

REQUIRED_PARAMS = ['CATGT_OUTPUT_DIR', 'OUTPUT_DIR', 'SESSION_ID', 'FAST_STORAGE_DIR']


def parse_params_file(params_file):
    try:
        params = DEFAULT_PARAMS.copy()

        with open(params_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            try:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()

                if key in params:
                    if key in ['PROBE_INDICES', 'EXTRACT_LFP_FOR', 'RUN_KILOSORT_FOR', 'PROCESS_BY_SHANK_FOR']:
                        if value.lower() == 'none':
                            params[key] = None
                        elif not value.strip():
                            params[key] = []
                        else:
                            params[key] = [int(x.strip()) for x in value.split(',')]
                    elif key in ['OVERRIDE_BAD_CHANNELS', 'OVERRIDE_LFP_CHANNELS']:
                        params[key] = value.lower() == 'true'
                    elif key in ['MAX_BAD_CHANNELS', 'LFP_SAMPLE_RATE', 'LFP_HIGH_PASS',
                                 'LFP_LOW_PASS', 'LFP_SPACING', 'INTER_SHANK_SPACE',
                                 'AP_HIGH_PASS', 'LFP_NUM_JOBS', 'SI_NUM_JOBS']:
                        params[key] = float(value)
                        if key in ['MAX_BAD_CHANNELS', 'LFP_SAMPLE_RATE', 'LFP_SPACING',
                                   'INTER_SHANK_SPACE', 'LFP_NUM_JOBS', 'SI_NUM_JOBS']:
                            params[key] = int(params[key])
                    elif key == 'KS_PARAMS':
                        params[key] = json.loads(value)
                    else:
                        params[key] = value
                else:
                    # Per-probe/shank manual bad channel file overrides, e.g.:
                    #   MANUAL_BAD_CHANNELS_FILE_IMEC0_SHANK0 = path/to/imec0_shank0_bad_channels.json
                    #   MANUAL_BAD_CHANNELS_FILE_IMEC0 = path/to/imec0_bad_channels.json
                    match = re.match(r'^MANUAL_BAD_CHANNELS_FILE_IMEC(\d+)(?:_SHANK(\d+))?$', key)
                    if match:
                        probe_num, shank_num = match.groups()
                        override_key = f"imec{probe_num}" if shank_num is None else f"imec{probe_num}_shank{shank_num}"
                        params['MANUAL_BAD_CHANNELS_FILES'][override_key] = value
                    else:
                        print(f"Warning: Unknown parameter '{key}' in params file")

            except Exception as e:
                print(f"Error parsing line '{line}': {e}")

        missing_params = [p for p in REQUIRED_PARAMS if params[p] is None]
        if missing_params:
            raise ValueError(f"Missing required parameters: {', '.join(missing_params)}")

        if params['EXTRACT_LFP_FOR'] is None:
            params['EXTRACT_LFP_FOR'] = params['PROBE_INDICES']
        if params['RUN_KILOSORT_FOR'] is None:
            params['RUN_KILOSORT_FOR'] = params['PROBE_INDICES']
        if params['PROCESS_BY_SHANK_FOR'] is None:
            params['PROCESS_BY_SHANK_FOR'] = []

        if params['OVERRIDE_BAD_CHANNELS'] and not params['MANUAL_BAD_CHANNELS_FILE'] and not params['MANUAL_BAD_CHANNELS_FILES']:
            raise ValueError(
                "OVERRIDE_BAD_CHANNELS is True but no manual bad channels file is configured. "
                "Set MANUAL_BAD_CHANNELS_FILE or per-probe/shank MANUAL_BAD_CHANNELS_FILE_IMEC*_SHANK* parameters."
            )

        return params

    except Exception as e:
        print(f"Error parsing parameters file '{params_file}': {e}")
        sys.exit(1)


def setup_logging(output_dir, session_id):
    log_file = os.path.join(
        output_dir,
        f"{session_id}_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def _load_manual_bad_channels(params, probe_idx, logger, shank_idx=None):
    tag = f"probe {probe_idx}" + (f" shank {shank_idx}" if shank_idx is not None else "")
    override_key = f"imec{probe_idx}_shank{shank_idx}" if shank_idx is not None else f"imec{probe_idx}"

    dedicated_file = params['MANUAL_BAD_CHANNELS_FILES'].get(override_key)
    manual_file = dedicated_file or params['MANUAL_BAD_CHANNELS_FILE']

    if not manual_file:
        logger.error(
            f"No manual bad channels file configured for {tag}. "
            f"Set MANUAL_BAD_CHANNELS_FILE or MANUAL_BAD_CHANNELS_FILE_{override_key.upper()}."
        )
        return None

    logger.info(f"Using manual bad channels for {tag} from: {manual_file}")

    try:
        with open(manual_file, 'r') as f:
            manual_data = json.load(f)

        if dedicated_file:
            # File dedicated to this probe/shank (e.g. produced by save_bad_channels): flat 'bad_channel_ids'
            if 'bad_channel_ids' in manual_data:
                return manual_data['bad_channel_ids']
            elif override_key in manual_data:
                return manual_data[override_key]['bad_channel_ids']
            else:
                logger.error(f"Could not find 'bad_channel_ids' in {manual_file} for {tag}")
                return None
        else:
            # Combined file covering multiple probes/shanks under nested keys
            shank_key = f"imec{probe_idx}_shank{shank_idx}" if shank_idx is not None else None
            if shank_key and shank_key in manual_data:
                return manual_data[shank_key]['bad_channel_ids']
            elif f'imec{probe_idx}' in manual_data:
                return manual_data[f'imec{probe_idx}']['bad_channel_ids']
            elif f'probe{probe_idx}' in manual_data:
                return manual_data[f'probe{probe_idx}']['bad_channel_ids']
            elif 'bad_channel_ids' in manual_data:
                return manual_data['bad_channel_ids']
            else:
                logger.error(f"Could not find bad channel data for {tag} in manual file")
                return None
    except Exception as e:
        logger.error(f"Error loading manual bad channels for {tag} from {manual_file}: {e}")
        return None


def detect_bad_channels(rec, probe_idx, logger, params, shank_idx=None):
    tag = f"probe {probe_idx}" + (f" shank {shank_idx}" if shank_idx is not None else "")
    logger.info(f"Detecting bad channels for {tag}...")

    try:
        bad_channel_ids, channel_labels = si.detect_bad_channels(rec)

        if isinstance(bad_channel_ids, np.ndarray):
            bad_channel_ids = bad_channel_ids.tolist()

        logger.info(f"Detected {len(bad_channel_ids)} bad channels for {tag}")
        for bad_channel_id in bad_channel_ids:
            try:
                index = int(bad_channel_id.split('AP')[1])
                logger.info(f"  {bad_channel_id}, {channel_labels[index]}")
            except (IndexError, ValueError):
                logger.info(f"  {bad_channel_id}")

        needs_manual_curation = len(bad_channel_ids) > params['MAX_BAD_CHANNELS']
        if needs_manual_curation:
            logger.warning(
                f"Bad channel count ({len(bad_channel_ids)}) exceeds threshold "
                f"({params['MAX_BAD_CHANNELS']}) for {tag}"
            )
            logger.warning("Manual curation is recommended")

        return bad_channel_ids, needs_manual_curation

    except Exception as e:
        logger.error(f"Exception detecting bad channels for {tag}: {e}")
        return [], True


def save_bad_channels(bad_channel_ids, catgt_output_dir, probe_idx, logger, shank_idx=None):
    try:
        fname = (
            f"imec{probe_idx}_shank{shank_idx}_bad_channels.json"
            if shank_idx is not None
            else f"imec{probe_idx}_bad_channels.json"
        )
        output_file = os.path.join(catgt_output_dir, fname)
        with open(output_file, 'w') as f:
            json.dump({'bad_channel_ids': bad_channel_ids}, f, indent=4)
        logger.info(f"Saved bad channel IDs to {output_file}")
        return True
    except Exception as e:
        logger.error(f"Exception saving bad channel IDs: {e}")
        return False


def save_lfp_times(final_tvec, probe_idx, catgt_output_dir, logger):
    """
    Save the LFP time vector for a probe.

    One file per probe, even when processing by shank: a probe's shanks are clocked
    together, so every shank yields the same time vector.
    """
    try:
        output_file = os.path.join(catgt_output_dir, f"imec{probe_idx}_lfp_times.npy")
        np.save(output_file, final_tvec)
        logger.info(f"Saved LFP times (.npy) to {output_file}")
        return True
    except Exception as e:
        logger.error(f"Exception saving LFP times for probe {probe_idx}: {e}")
        return False


def extract_lfp(rec, bad_channel_ids, probe_idx, logger, params, shank_idx=None):
    tag = f"probe {probe_idx}" + (f" shank {shank_idx}" if shank_idx is not None else "")
    try:
        logger.info(f"Extracting LFP for {tag}...")

        lfp_rec = rec.remove_channels(bad_channel_ids)

        # This is not needed because this is an erroneous check: https://github.com/SpikeInterface/spikeinterface/pull/4506
        # Also see https://github.com/SpikeInterface/spikeinterface/issues/4469
        # si.set_global_job_kwargs(chunk_duration=params['LFP_CHUNK_DURATION'])
        # logger.info(f"Changed global job kwargs for lfp filtering - increased chunk duration to {params['LFP_CHUNK_DURATION']}")
        # logger.info(f"Global job kwargs are now: {si.get_global_job_kwargs()}")
        
        logger.info(f"Applying bandpass filter ({params['LFP_HIGH_PASS']}-{params['LFP_LOW_PASS']} Hz)...")
        margin_ms_lfp = params['LFP_HIGH_PASS']*5*1000 # high-pass freq threshold * 5, expressed in ms (e.g. 1Hz --> 5000 ms)
        lfp_rec = si.bandpass_filter(lfp_rec, freq_min=params['LFP_HIGH_PASS'], freq_max=params['LFP_LOW_PASS'], \
                                     margin_ms=margin_ms_lfp, ignore_low_freq_error=True)
        
        logger.info(f"Resampling to {params['LFP_SAMPLE_RATE']} Hz...")
        lfp_rec = si.resample(lfp_rec, params['LFP_SAMPLE_RATE'], margin_ms=margin_ms_lfp)

        if params['OVERRIDE_LFP_CHANNELS']:
            logger.info(f"Using manual LFP channel selection from: {params['MANUAL_LFP_CHANNELS_FILE']}")
            try:
                with open(params['MANUAL_LFP_CHANNELS_FILE'], 'r') as f:
                    manual_data = json.load(f)

                shank_key = f"imec{probe_idx}_shank{shank_idx}" if shank_idx is not None else None
                if shank_key and shank_key in manual_data:
                    manual_channels = manual_data[shank_key]['channel_ids']
                elif f'imec{probe_idx}' in manual_data:
                    manual_channels = manual_data[f'imec{probe_idx}']['channel_ids']
                elif f'probe{probe_idx}' in manual_data:
                    manual_channels = manual_data[f'probe{probe_idx}']['channel_ids']
                elif 'channel_ids' in manual_data:
                    manual_channels = manual_data['channel_ids']
                else:
                    logger.error(f"Could not find LFP channel data for {tag} in manual file")
                    return None, None, None, None, None, None

                if not isinstance(manual_channels, np.ndarray):
                    manual_channels = np.array(manual_channels)

                available_channels = set(lfp_rec.channel_ids)
                valid_channels = [ch for ch in manual_channels if ch in available_channels]

                if len(valid_channels) < len(manual_channels):
                    logger.warning(
                        f"{len(manual_channels) - len(valid_channels)} manual LFP channels "
                        f"not found for {tag}"
                    )
                if len(valid_channels) == 0:
                    logger.error(f"No valid LFP channels found in manual selection for {tag}")
                    return None, None, None, None, None, None

                final_channels = np.array(valid_channels)
                logger.info(f"Selected {len(final_channels)} manual LFP channels for {tag}")

                channel_indices = []
                for ch in final_channels:
                    try:
                        idx = np.where(lfp_rec.channel_ids == ch)[0][0]
                        channel_indices.append(idx)
                    except (IndexError, ValueError):
                        logger.warning(f"Channel {ch} not found in recording")

                all_locs = lfp_rec.get_channel_locations()
                final_depths = [(all_locs[idx][1] + 175) for idx in channel_indices]
                shank_ids = [int(all_locs[idx][0] // params['INTER_SHANK_SPACE']) for idx in channel_indices]

            except Exception as e:
                logger.error(f"Error loading manual LFP channels for {tag}: {e}")
                return None, None, None, None, None, None

        else:
            logger.info(f"Selecting channels with {params['LFP_SPACING']} um spacing for {tag}...")
            all_locs = lfp_rec.get_channel_locations()
            keep_idx = []

            if shank_idx is not None:
                # Per-shank: recording already contains only this shank's channels
                idx_depth = [(i, loc[1]) for i, loc in enumerate(all_locs)]
                if len(idx_depth) == 0:
                    logger.error(f"No channels found for {tag}")
                    return None, None, None, None, None, None

                depths = [x[1] for x in idx_depth]
                idx    = [x[0] for x in idx_depth]

                depths_array  = np.array(depths)
                sorted_depths = np.sort(depths_array)
                depth_diffs   = np.diff(sorted_depths)
                gap_indices   = np.where(np.abs(depth_diffs) > 100)[0]

                if len(gap_indices) > 0:
                    queried_depths = []
                    start = 0
                    for gap_idx in gap_indices:
                        seg = sorted_depths[start:gap_idx + 1]
                        queried_depths.extend(np.arange(min(seg), max(seg), params['LFP_SPACING']))
                        start = gap_idx + 1
                    if start < len(sorted_depths):
                        seg = sorted_depths[start:]
                        queried_depths.extend(np.arange(min(seg), max(seg), params['LFP_SPACING']))
                else:
                    queried_depths = np.arange(min(depths), max(depths), params['LFP_SPACING']).tolist()

                if queried_depths[-1] < max(depths) - 50:
                    queried_depths.append(max(depths))

                for b in queried_depths:
                    min_diff = float('inf')
                    closest_index = None
                    for i, a in enumerate(depths):
                        diff = abs(a - b)
                        if diff < min_diff:
                            min_diff = diff
                            closest_index = i
                        elif diff == min_diff:
                            closest_index = min(closest_index, i)
                    if idx[closest_index] not in keep_idx:
                        keep_idx.append(idx[closest_index])

                shank_ids = [shank_idx] * len(keep_idx)

            else:
                # Per-probe: loop over shanks to sample channels from each
                for iShank in range(4):
                    idx_depth = [
                        (i, loc[1]) for i, loc in enumerate(all_locs)
                        if int(loc[0] // params['INTER_SHANK_SPACE']) == iShank
                    ]
                    if not idx_depth:
                        continue

                    depths = [x[1] for x in idx_depth]
                    idx    = [x[0] for x in idx_depth]

                    depths_array  = np.array(depths)
                    sorted_depths = np.sort(depths_array)
                    depth_diffs   = np.diff(sorted_depths)
                    gap_indices   = np.where(np.abs(depth_diffs) > 100)[0]

                    if len(gap_indices) > 0:
                        queried_depths = []
                        start_idx = 0
                        for gap_idx in gap_indices:
                            seg = sorted_depths[start_idx:gap_idx + 1]
                            queried_depths.extend(np.arange(min(seg), max(seg), params['LFP_SPACING']))
                            start_idx = gap_idx + 1
                        if start_idx < len(sorted_depths):
                            seg = sorted_depths[start_idx:]
                            queried_depths.extend(np.arange(min(seg), max(seg), params['LFP_SPACING']))
                    else:
                        queried_depths = np.arange(min(depths), max(depths), params['LFP_SPACING']).tolist()

                    if queried_depths[-1] < max(depths) - 50:
                        queried_depths.append(max(depths))

                    for b in queried_depths:
                        min_diff = float('inf')
                        closest_index = None
                        for i, a in enumerate(depths):
                            diff = abs(a - b)
                            if diff < min_diff:
                                min_diff = diff
                                closest_index = i
                            elif diff == min_diff:
                                closest_index = min(closest_index, i)
                        if idx[closest_index] not in keep_idx:
                            keep_idx.append(idx[closest_index])

                shank_ids = [int(all_locs[i][0] // params['INTER_SHANK_SPACE']) for i in keep_idx]

            final_channels = lfp_rec.channel_ids[np.asarray(keep_idx)]
            logger.info(f"Selected {len(final_channels)} LFP channels automatically for {tag}")
            final_depths = [(all_locs[i][1] + 175) for i in keep_idx]

        # Write to temp binary for fast extraction
        temp_suffix = f"imec{probe_idx}" if shank_idx is None else f"imec{probe_idx}_shank{shank_idx}"
        temp_lfp_folder = os.path.join(
            params['FAST_STORAGE_DIR'],
            f"{params['SESSION_ID']}_temp_lfp_{temp_suffix}"
        )
        os.makedirs(temp_lfp_folder, exist_ok=True)

        logger.info(f"Creating temporary binary for LFP extraction ({tag})...")
        job_kwargs = dict(n_jobs=params['LFP_NUM_JOBS'], chunk_duration=params['LFP_CHUNK_DURATION'], progress_bar=True)
        temp_lfp = lfp_rec.save(
            folder=temp_lfp_folder,
            channel_ids=final_channels,
            format="binary",
            return_scaled=True,
            overwrite=True,
            **job_kwargs
        )

        logger.info("Extracting LFP traces from temporary file...")
        final_lfp = temp_lfp.get_traces(channel_ids=final_channels, return_scaled=True)

        final_tvec = lfp_rec.get_times()
        final_tvec = final_tvec - final_tvec[0]
        final_fs   = lfp_rec.get_sampling_frequency()

        logger.info("Cleaning up temporary LFP files...")
        try:
            shutil.rmtree(temp_lfp_folder)
        except Exception as e:
            logger.warning(f"Could not remove temporary LFP folder {temp_lfp_folder}: {e}")

        return final_channels, final_lfp, final_tvec, final_fs, final_depths, shank_ids

    except Exception as e:
        logger.error(f"Exception extracting LFP for {tag}: {e}")
        return None, None, None, None, None, None


def run_kilosort(rec, bad_channel_ids, probe_idx, logger, params, shank_idx=None):
    tag = f"probe {probe_idx}" + (f" shank {shank_idx}" if shank_idx is not None else "")
    try:
        logger.info(f"Running Kilosort4 for {tag}...")

        logger.info(f"Applying high-pass filter at {params['AP_HIGH_PASS']} Hz...")
        filt_rec = si.highpass_filter(rec, freq_min=params['AP_HIGH_PASS'])

        logger.info(f"Removing {len(bad_channel_ids)} bad channels...")
        filt_rec = filt_rec.remove_channels(bad_channel_ids)

        folder_name = (
            f"{params['SESSION_ID']}_imec{probe_idx}_shank{shank_idx}"
            if shank_idx is not None
            else f"{params['SESSION_ID']}_imec{probe_idx}"
        )
        si_working_folder = os.path.join(params['FAST_STORAGE_DIR'], f"{folder_name}_si_preprocess")
        ks_working_folder = os.path.join(params['FAST_STORAGE_DIR'], folder_name)

        os.makedirs(si_working_folder, exist_ok=True)
        os.makedirs(ks_working_folder, exist_ok=True)

        logger.info(f"Saving filtered recording to {si_working_folder}...")
        job_kwargs = dict(n_jobs=params['SI_NUM_JOBS'], chunk_duration=params['CHUNK_DURATION'], progress_bar=True)
        saved_rec = filt_rec.save(folder=si_working_folder, format='binary', overwrite=True, **job_kwargs)

        logger.info(f"Running Kilosort4 with output to {ks_working_folder}...")
        si.run_sorter(
            'kilosort4', saved_rec,
            folder=ks_working_folder,
            verbose=True,
            remove_existing_folder=True,
            **params['KS_PARAMS']
        )

        logger.info(f"Kilosort4 completed for {tag}")
        return True

    except Exception as e:
        logger.error(f"Exception running Kilosort for {tag}: {e}")
        return False


def process_probe(probe_idx, logger, params):
    logger.info(f"Processing probe {probe_idx}...")

    try:
        logger.info(f"Reading raw data from {params['CATGT_OUTPUT_DIR']} for stream imec{probe_idx}.ap...")
        raw_rec = si.read_spikeglx(params['CATGT_OUTPUT_DIR'], stream_name=f"imec{probe_idx}.ap")

        if probe_idx in params['PROCESS_BY_SHANK_FOR']:
            logger.info("Splitting recording by channel group (shank)...")
            shank_recs = raw_rec.split_by("group")
            logger.info(f"Found {len(shank_recs)} shank(s): {sorted(shank_recs.keys())}")

            all_success = True
            probe_lfp_tvec = None  # shanks are clock-locked; one time vector per probe
            for shank_idx, shank_rec in sorted(shank_recs.items()):
                tag = f"probe {probe_idx} shank {shank_idx}"
                logger.info(f"--- Starting {tag} ({shank_rec.get_num_channels()} channels) ---")

                if params['OVERRIDE_BAD_CHANNELS']:
                    bad_channel_ids = _load_manual_bad_channels(
                        params, probe_idx, logger, shank_idx=shank_idx
                    )
                    if bad_channel_ids is None:
                        all_success = False
                        continue
                    needs_manual_curation = False
                else:
                    bad_channel_ids, needs_manual_curation = detect_bad_channels(
                        shank_rec, probe_idx, logger, params, shank_idx=shank_idx
                    )

                save_bad_channels(
                    bad_channel_ids, params['CATGT_OUTPUT_DIR'], probe_idx, logger, shank_idx=shank_idx
                )

                if needs_manual_curation:
                    logger.warning(f"Skipping {tag} due to high bad channel count or detection error")
                    all_success = False
                    continue

                if probe_idx in params['EXTRACT_LFP_FOR']:
                    final_channels, final_lfp, final_tvec, final_fs, final_depths, shank_ids = extract_lfp(
                        shank_rec, bad_channel_ids, probe_idx, logger, params, shank_idx=shank_idx
                    )
                    if final_lfp is None:
                        logger.error(f"LFP extraction failed for {tag}")
                        all_success = False
                    else:
                        lfp_mat_fname = os.path.join(
                            params['OUTPUT_DIR'],
                            f"imec{probe_idx}_shank{shank_idx}_clean_lfp.mat"
                        )
                        logger.info(f"Saving LFP data to {lfp_mat_fname}")
                        scio.savemat(lfp_mat_fname, {
                            'depths':      final_depths,
                            'channel_ids': final_channels,
                            'lfp_traces':  np.squeeze(np.asarray(final_lfp)),
                            'lfp_tvec':    final_tvec,
                            'lfp_fs':      final_fs,
                            'shank_ids':   shank_ids,
                        })

                        if probe_lfp_tvec is None:
                            probe_lfp_tvec = final_tvec
                        elif (len(final_tvec) != len(probe_lfp_tvec)
                              or not np.allclose(final_tvec, probe_lfp_tvec)):
                            logger.warning(
                                f"LFP time vector for {tag} differs from earlier shanks on probe "
                                f"{probe_idx}; shanks are expected to be clock-locked"
                            )

                        logger.info(f"LFP extraction completed for {tag}")
                else:
                    logger.info(f"Skipping LFP extraction for {tag} (not in EXTRACT_LFP_FOR)")

                if probe_idx in params['RUN_KILOSORT_FOR']:
                    success = run_kilosort(
                        shank_rec, bad_channel_ids, probe_idx, logger, params, shank_idx=shank_idx
                    )
                    if not success:
                        logger.error(f"Kilosort failed for {tag}")
                        all_success = False
                    else:
                        logger.info(f"Kilosort completed for {tag}")
                else:
                    logger.info(f"Skipping Kilosort for {tag} (not in RUN_KILOSORT_FOR)")

                logger.info(f"--- Finished {tag} ---")

            if probe_lfp_tvec is not None:
                save_lfp_times(probe_lfp_tvec, probe_idx, params['CATGT_OUTPUT_DIR'], logger)

            return all_success

        else:
            # Per-probe mode
            if params['OVERRIDE_BAD_CHANNELS']:
                bad_channel_ids = _load_manual_bad_channels(
                    params, probe_idx, logger
                )
                if bad_channel_ids is None:
                    return False
                needs_manual_curation = False
            else:
                bad_channel_ids, needs_manual_curation = detect_bad_channels(
                    raw_rec, probe_idx, logger, params
                )

            save_bad_channels(bad_channel_ids, params['CATGT_OUTPUT_DIR'], probe_idx, logger)

            if needs_manual_curation:
                logger.warning(f"Skipping probe {probe_idx} due to high bad channel count or detection error")
                return False

            if probe_idx in params['EXTRACT_LFP_FOR']:
                final_channels, final_lfp, final_tvec, final_fs, final_depths, shank_ids = extract_lfp(
                    raw_rec, bad_channel_ids, probe_idx, logger, params
                )
                if final_lfp is None:
                    logger.error(f"LFP extraction failed for probe {probe_idx}")
                    return False

                save_lfp_times(final_tvec, probe_idx, params['CATGT_OUTPUT_DIR'], logger)

                lfp_mat_fname = os.path.join(params['OUTPUT_DIR'], f"imec{probe_idx}_clean_lfp.mat")
                logger.info(f"Saving LFP data to {lfp_mat_fname}")
                scio.savemat(lfp_mat_fname, {
                    'depths':      final_depths,
                    'channel_ids': final_channels,
                    'lfp_traces':  np.squeeze(np.asarray(final_lfp)),
                    'lfp_tvec':    final_tvec,
                    'lfp_fs':      final_fs,
                    'shank_ids':   shank_ids,
                })
                logger.info(f"LFP data extraction completed for probe {probe_idx}")
            else:
                logger.info(f"Skipping LFP extraction for probe {probe_idx} (not in EXTRACT_LFP_FOR list)")

            if probe_idx in params['RUN_KILOSORT_FOR']:
                success = run_kilosort(raw_rec, bad_channel_ids, probe_idx, logger, params)
                if not success:
                    logger.error(f"Kilosort processing failed for probe {probe_idx}")
                    return False
                logger.info(f"Kilosort processing completed for probe {probe_idx}")
            else:
                logger.info(f"Skipping Kilosort for probe {probe_idx} (not in RUN_KILOSORT_FOR list)")

            logger.info(f"Successfully processed probe {probe_idx}")
            return True

    except Exception as e:
        logger.error(f"Exception processing probe {probe_idx}: {e}")
        return False


def process_session(params_file):
    params = parse_params_file(params_file)
    logger = setup_logging(params['OUTPUT_DIR'], params['SESSION_ID'])

    logger.info(f"Processing session: {params['SESSION_ID']}")
    logger.info(f"Parameters file: {params_file}")
    logger.info(f"CatGT output directory: {params['CATGT_OUTPUT_DIR']}")
    logger.info(f"Output directory: {params['OUTPUT_DIR']}")
    logger.info(f"Fast storage for KS4: {params['FAST_STORAGE_DIR']}")
    logger.info(f"Processing probes: {params['PROBE_INDICES']}")
    logger.info(f"Extracting LFP for probes: {params['EXTRACT_LFP_FOR']}")
    logger.info(f"Running Kilosort for probes: {params['RUN_KILOSORT_FOR']}")
    logger.info(f"Process by shank for probes: {params['PROCESS_BY_SHANK_FOR']}")

    os.makedirs(params['OUTPUT_DIR'], exist_ok=True)
    os.makedirs(params['FAST_STORAGE_DIR'], exist_ok=True)

    all_success = True
    for probe_idx in params['PROBE_INDICES']:
        success = process_probe(probe_idx, logger, params)
        if not success:
            all_success = False
            logger.warning(f"Processing failed or was skipped for probe {probe_idx}")
        else:
            logger.info(f"Processing completed successfully for probe {probe_idx}")

    if all_success:
        logger.info(f"All probes processed successfully for session {params['SESSION_ID']}")
        print(f"\nAll probes processed successfully for session {params['SESSION_ID']}!")
    else:
        logger.warning(f"Processing completed with warnings or errors for session {params['SESSION_ID']}")
        print(f"\nProcessing completed with warnings or errors for session {params['SESSION_ID']}. Check the log file for details.")

    return all_success


def main():
    parser = argparse.ArgumentParser(description='Process Neuropixels data: extract LFP and run Kilosort')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--file_params', type=str, help='Path to parameters file for a single session')
    group.add_argument('--batch_params', type=str, help='Path to a file listing multiple parameter files')
    args = parser.parse_args()

    if args.file_params:
        process_session(args.file_params)
    else:
        with open(args.batch_params, 'r') as f:
            params_files = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]

        print(f"Processing {len(params_files)} sessions from batch file: {args.batch_params}")
        success_count = 0
        for i, params_file in enumerate(params_files):
            print(f"\nProcessing session {i+1}/{len(params_files)}: {params_file}")
            if process_session(params_file):
                success_count += 1

        print(f"\nBatch processing completed: {success_count}/{len(params_files)} sessions processed successfully")


if __name__ == "__main__":
    main()
