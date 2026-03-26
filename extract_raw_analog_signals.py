#!/usr/bin/env python3
"""
Extract Raw Analog Signals from Neuropixels NIDQ Stream

This script extracts full timeseries data from specified analog channels in the
NIDQ stream of a Neuropixels recording. Each channel is bandpass-filtered,
downsampled, and saved as an individual .mat file (data, tvec, fs) plus a .npy
time vector in CATGT_OUTPUT_DIR for subsequent TPrime alignment.

RAW_ANALOG_CHANNELS is a JSON dict mapping channel name -> [high_pass, low_pass, sample_rate].
Any value omitted from the list falls back to the global default.

Example:
    RAW_ANALOG_CHANNELS = {"XA0": [0.1, 500, 1000], "XA7": [0.1, 500, 2500]}

Defaults:  bandpass 0.1-500 Hz  ->  downsample to 1000 Hz

Usage:
    python extract_raw_analog_signals.py --file_params path/to/params.txt
    python extract_raw_analog_signals.py --batch_params path/to/batch_list.txt
"""

import os
import sys
import json
import logging
import numpy as np
import scipy.io as scio
import argparse
from collections import defaultdict
from datetime import datetime
import spikeinterface.full as si


# ── Default parameters ────────────────────────────────────────────────────────

DEFAULT_PARAMS = {
    # Required
    'SOURCE_DIR': None,          # original recording directory (contains .nidq.bin/.meta)
    'CATGT_OUTPUT_DIR': None,    # .npy time vectors are written here
    'OUTPUT_DIR': None,          # destination for .mat files
    'SESSION_ID': None,
    # JSON dict: {"XA0": [high_pass, low_pass, sample_rate], "XA7": [...], ...}
    # Any value absent from the list falls back to the global defaults below.
    'RAW_ANALOG_CHANNELS': None,

    # Global defaults - used when a channel's list is missing a value
    'RAW_ANALOG_DEFAULT_HIGH_PASS': 0.1,    # Hz
    'RAW_ANALOG_DEFAULT_LOW_PASS': 500.0,   # Hz
    'RAW_ANALOG_DEFAULT_SAMPLE_RATE': 1000, # Hz

    'NUM_JOBS': 4,
    'CHUNK_DURATION': '1s'
}

REQUIRED_PARAMS = ['SOURCE_DIR', 'CATGT_OUTPUT_DIR', 'OUTPUT_DIR', 'SESSION_ID', 'RAW_ANALOG_CHANNELS']


# ── Parameter parsing ─────────────────────────────────────────────────────────

def parse_params_file(params_file):
    """Parse parameters from a file"""
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

                if key not in params:
                    print(f"Warning: Unknown parameter '{key}' in params file")
                    continue

                if key == 'RAW_ANALOG_CHANNELS':
                    params[key] = json.loads(value)
                elif key == 'RAW_ANALOG_DEFAULT_SAMPLE_RATE':
                    params[key] = int(value)
                elif key in ['RAW_ANALOG_DEFAULT_HIGH_PASS', 'RAW_ANALOG_DEFAULT_LOW_PASS']:
                    params[key] = float(value)
                elif key == 'NUM_JOBS':
                    params[key] = int(value)
                else:
                    params[key] = value

            except Exception as e:
                print(f"Error parsing line '{line}': {e}")

        missing = [p for p in REQUIRED_PARAMS if params[p] is None]
        if missing:
            raise ValueError(f"Missing required parameters: {', '.join(missing)}")

        return params

    except Exception as e:
        print(f"Error parsing parameters file '{params_file}': {e}")
        sys.exit(1)


def _channel_settings(ch_name, ch_config, params):
    """
    Resolve [high_pass, low_pass, sample_rate] for one channel.

    ch_config is the list from the JSON dict (may have 0-3 elements).
    Missing values fall back to the global defaults.
    """
    high_pass = float(ch_config[0]) if len(ch_config) > 0 else params['RAW_ANALOG_DEFAULT_HIGH_PASS']
    low_pass  = float(ch_config[1]) if len(ch_config) > 1 else params['RAW_ANALOG_DEFAULT_LOW_PASS']
    target_fs = int(ch_config[2])   if len(ch_config) > 2 else params['RAW_ANALOG_DEFAULT_SAMPLE_RATE']
    return high_pass, low_pass, target_fs


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging(output_dir, session_id):
    log_file = os.path.join(
        output_dir,
        f"{session_id}_raw_analog_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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


# ── Helpers ───────────────────────────────────────────────────────────────────

def resolve_channel_id(channel_name, available_ids):
    """
    Map a user channel name (e.g. 'XA0') to the SpikeInterface channel ID
    (e.g. 'nidq#XA0').  Returns None if not found.
    """
    if channel_name in available_ids:
        return channel_name
    for prefix in ('nidq#', 'nidq.'):
        candidate = f"{prefix}{channel_name}"
        if candidate in available_ids:
            return candidate
    # Partial match as last resort
    for ch_id in available_ids:
        if channel_name in ch_id:
            return ch_id
    return None


def apply_filter(raw_rec, high_pass, low_pass, logger):
    """Apply bandpass / high-pass / low-pass filter to a recording."""
    hp = high_pass > 0
    lp = low_pass > 0

    # Clamp low-pass to below Nyquist of the recording
    nyquist = raw_rec.get_sampling_frequency() / 2.0
    if lp and low_pass >= nyquist:
        logger.warning(
            f"Low-pass cutoff {low_pass} Hz >= Nyquist {nyquist} Hz - "
            f"clamping to {nyquist * 0.95:.1f} Hz"
        )
        low_pass = nyquist * 0.95

    if hp and lp:
        logger.info(f"  Applying bandpass filter ({high_pass}-{low_pass} Hz)...")
        return si.bandpass_filter(raw_rec, freq_min=high_pass, freq_max=low_pass)
    elif hp:
        logger.info(f"  Applying high-pass filter at {high_pass} Hz...")
        return si.highpass_filter(raw_rec, freq_min=high_pass)
    elif lp:
        logger.info(f"  Applying low-pass filter at {low_pass} Hz...")
        return si.bandpass_filter(raw_rec, freq_min=0.1, freq_max=low_pass)
    else:
        return raw_rec


# ── Core extraction ───────────────────────────────────────────────────────────

def extract_single_channel(resampled_rec, channel_name, channel_id,
                            target_fs, params, logger):
    """
    Extract traces for one channel from an already-filtered+resampled recording,
    save .mat to OUTPUT_DIR and .npy tvec to CATGT_OUTPUT_DIR.
    """
    logger.info(f"  Extracting '{channel_name}' (id: {channel_id}) at {target_fs} Hz...")
    try:
        # Correct for SpikeInterface t_start offset (same fix as in extract_lfp_and_run_ks4.py)
        tvec = resampled_rec.get_times()
        tvec = tvec - tvec[0]

        traces = resampled_rec.get_traces(channel_ids=[channel_id], return_scaled=True)
        data = np.squeeze(np.asarray(traces))

        # .mat - data + time vector + metadata
        mat_fname = os.path.join(params['OUTPUT_DIR'], f"{channel_name}_unsynced.mat")
        scio.savemat(mat_fname, {
            'data': data,
            'tvec': tvec,
            'fs': float(target_fs),
            'channel_id': channel_id,
        })
        logger.info(f"  Saved .mat -> {mat_fname}")

        # .npy tvec -> CATGT_OUTPUT_DIR for TPrime
        times_npy_path = os.path.join(
            params['CATGT_OUTPUT_DIR'], f"{channel_name}_raw_times.npy"
        )
        np.save(times_npy_path, tvec)
        logger.info(f"  Saved time vector (.npy) -> {times_npy_path}")

        return True

    except Exception as e:
        logger.error(f"  Exception extracting '{channel_name}': {e}")
        return False


def extract_raw_analog_signals(params, logger):
    """
    Main extraction routine.

    Groups channels by their unique (high_pass, low_pass, target_fs) combination
    to avoid redundant filter/resample operations.
    """
    logger.info(f"Reading NIDQ stream from {params['SOURCE_DIR']}...")
    try:
        raw_rec = si.read_spikeglx(params['SOURCE_DIR'], stream_name='nidq')
    except Exception as e:
        logger.error(f"Failed to read NIDQ stream: {e}")
        return False

    native_fs = raw_rec.get_sampling_frequency()
    available_ids = raw_rec.channel_ids.tolist()
    logger.info(f"Native sampling rate: {native_fs} Hz")
    logger.info(f"Available NIDQ channel IDs: {available_ids}")

    # Resolve user channel names -> SpikeInterface IDs
    channel_map = {}   # ch_name -> ch_id
    for ch_name in params['RAW_ANALOG_CHANNELS']:
        ch_id = resolve_channel_id(ch_name, available_ids)
        if ch_id is None:
            logger.error(f"Channel '{ch_name}' not found in NIDQ stream - skipping")
        else:
            channel_map[ch_name] = ch_id
            logger.info(f"Resolved '{ch_name}' -> '{ch_id}'")

    if not channel_map:
        logger.error("No valid channels found. Aborting.")
        return False

    # Group channels by unique (high_pass, low_pass, target_fs) to share filter/resample
    groups = defaultdict(list)   # (hp, lp, fs) -> [channel_names]
    for ch_name in channel_map:
        ch_config = params['RAW_ANALOG_CHANNELS'][ch_name]
        settings = _channel_settings(ch_name, ch_config, params)
        groups[settings].append(ch_name)

    all_success = True

    for (high_pass, low_pass, target_fs), channels in groups.items():
        logger.info(
            f"Processing {channels} - "
            f"bandpass {high_pass}-{low_pass} Hz, downsample to {target_fs} Hz"
        )

        try:
            filtered_rec = apply_filter(raw_rec, high_pass, low_pass, logger)
        except Exception as e:
            logger.error(f"Filtering failed for {channels}: {e}")
            all_success = False
            continue

        try:
            logger.info(f"  Resampling from {native_fs} Hz to {target_fs} Hz...")
            resampled_rec = si.resample(filtered_rec, target_fs)
        except Exception as e:
            logger.error(f"Resampling to {target_fs} Hz failed for {channels}: {e}")
            all_success = False
            continue

        for ch_name in channels:
            ch_id = channel_map[ch_name]
            success = extract_single_channel(
                resampled_rec, ch_name, ch_id, target_fs, params, logger
            )
            if not success:
                all_success = False

    return all_success


# ── Session / batch wrappers ──────────────────────────────────────────────────

def process_session(params_file):
    """Process a single session"""
    params = parse_params_file(params_file)
    os.makedirs(params['OUTPUT_DIR'], exist_ok=True)
    logger = setup_logging(params['OUTPUT_DIR'], params['SESSION_ID'])

    logger.info(f"Processing session: {params['SESSION_ID']}")
    logger.info(f"Parameters file: {params_file}")
    logger.info(f"Source directory (nidq): {params['SOURCE_DIR']}")
    logger.info(f"CatGT output directory (npy times): {params['CATGT_OUTPUT_DIR']}")
    logger.info(f"Output directory (mat files): {params['OUTPUT_DIR']}")
    logger.info(
        f"Global defaults - HP: {params['RAW_ANALOG_DEFAULT_HIGH_PASS']} Hz, "
        f"LP: {params['RAW_ANALOG_DEFAULT_LOW_PASS']} Hz, "
        f"sample rate: {params['RAW_ANALOG_DEFAULT_SAMPLE_RATE']} Hz"
    )
    for ch_name, ch_config in params['RAW_ANALOG_CHANNELS'].items():
        hp, lp, fs = _channel_settings(ch_name, ch_config, params)
        logger.info(f"  {ch_name}: HP={hp} Hz, LP={lp} Hz, fs={fs} Hz")

    success = extract_raw_analog_signals(params, logger)

    if success:
        logger.info(f"Extraction completed successfully for session {params['SESSION_ID']}")
        print(f"\nAnalog extraction completed successfully for session {params['SESSION_ID']}!")
    else:
        logger.warning(f"Extraction completed with errors for session {params['SESSION_ID']}")
        print(f"\nAnalog extraction completed with errors for {params['SESSION_ID']}. Check log for details.")

    return success


def main():
    parser = argparse.ArgumentParser(
        description='Extract raw analog signals from Neuropixels NIDQ stream'
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--file_params', type=str,
                       help='Path to parameters file for a single session')
    group.add_argument('--batch_params', type=str,
                       help='Path to a file listing multiple parameter files')
    args = parser.parse_args()

    if args.file_params:
        process_session(args.file_params)
    else:
        with open(args.batch_params, 'r') as f:
            params_files = [
                line.strip() for line in f
                if line.strip() and not line.strip().startswith('#')
            ]
        print(f"Processing {len(params_files)} sessions from batch file: {args.batch_params}")
        success_count = 0
        for i, params_file in enumerate(params_files):
            print(f"\nProcessing session {i+1}/{len(params_files)}: {params_file}")
            if process_session(params_file):
                success_count += 1
        print(f"\nBatch processing completed: {success_count}/{len(params_files)} sessions processed successfully")


if __name__ == "__main__":
    main()
