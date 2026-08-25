#!/usr/bin/env python3
"""
Extract Units After Phy Curation

This script extracts curated units (both good and MUA) after manual Phy curation.
It processes Kilosort output folders and generates MATLAB-compatible files with
spike times, waveforms, and metadata for further analysis.

Usage:
    python extract_units_post_curation.py --file_params path/to/params.txt
    python extract_units_post_curation.py --batch_params path/to/batch_list.txt
"""

import os
import sys
import json
import logging
import numpy as np
import scipy.io as scio
import argparse
from datetime import datetime
import spikeinterface.full as si

# Default parameters
DEFAULT_PARAMS = {
    'CATGT_OUTPUT_DIR': None,  # Required - path to CatGT output with tcat raw recordings
    'OUTPUT_DIR': None,        # Required - path to save extracted unit files
    'SESSION_ID': None,        # Required - session identifier
    'PROBE_INDICES': [0],      # List of probe indices to process
    'PROCESSED_BY_SHANK': False, # If True, expect per-shank KS folders and merge into one per-probe .mat
    'SHANK_INDICES': None,       # Required when PROCESSED_BY_SHANK=True (e.g. 0,1,2,3)
    'SORTER_OUTPUT_FOLDERS': {}, # Per-probe: {"0": "path"} or per-shank: {"0_shank0": "path", "0_shank1": "path", ...}
    'PREPROCESS_FOLDERS': {},  # Same key format as SORTER_OUTPUT_FOLDERS
    'SAVE_WAVEFORMS': True,    # Whether to extract and save unit waveforms
    'EXTRACT_GOOD_UNITS': True, # Whether to extract good units
    'EXTRACT_MUA_UNITS': True,  # Whether to extract MUA units
    'NUM_JOBS': 8,             # Number of parallel jobs for waveform extraction
}

# Required parameters
REQUIRED_PARAMS = ['CATGT_OUTPUT_DIR', 'OUTPUT_DIR', 'SESSION_ID']

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
                
                if key in params:
                    # Handle different parameter types
                    if key in ['PROBE_INDICES', 'SHANK_INDICES']:
                        # Parse list of integers
                        if value.lower() == 'none' or not value.strip():
                            params[key] = None
                        else:
                            params[key] = [int(x.strip()) for x in value.split(',')]
                    elif key in ['SAVE_WAVEFORMS', 'EXTRACT_GOOD_UNITS', 'EXTRACT_MUA_UNITS', 'PROCESSED_BY_SHANK']:
                        # Parse boolean
                        params[key] = value.lower() == 'true'
                    elif key == 'NUM_JOBS':
                        # Parse integer
                        params[key] = int(value)
                    elif key in ['SORTER_OUTPUT_FOLDERS', 'PREPROCESS_FOLDERS']:
                        # Parse JSON dictionary mapping probe indices to folder paths
                        params[key] = json.loads(value)
                    else:
                        # String parameters
                        params[key] = value
                else:
                    print(f"Warning: Unknown parameter '{key}' in params file")
            except Exception as e:
                print(f"Error parsing line '{line}': {e}")
        
        # Check required parameters
        missing_params = [param for param in REQUIRED_PARAMS if params[param] is None]
        if missing_params:
            raise ValueError(f"Missing required parameters: {', '.join(missing_params)}")
        
        return params
        
    except Exception as e:
        print(f"Error parsing parameters file '{params_file}': {e}")
        sys.exit(1)

# Set up logging
def setup_logging(output_dir, session_id):
    log_file = os.path.join(output_dir, f"{session_id}_extract_units_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # Reset the root logger and remove all handlers
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

def load_bad_channels(catgt_output_dir, probe_idx, logger, shank_idx=None):
    """
    Load bad channel IDs from a JSON file.

    Reads imec{p}_shank{s}_bad_channels.json when shank_idx is given,
    otherwise falls back to the legacy imec{p}_bad_channels.json.
    """
    try:
        if shank_idx is not None:
            bad_channels_file = os.path.join(
                catgt_output_dir, f"imec{probe_idx}_shank{shank_idx}_bad_channels.json"
            )
        else:
            bad_channels_file = os.path.join(catgt_output_dir, f"imec{probe_idx}_bad_channels.json")

        if not os.path.isfile(bad_channels_file):
            label = f"probe {probe_idx}" if shank_idx is None else f"probe {probe_idx} shank {shank_idx}"
            logger.warning(f"Bad channels file not found for {label}: {bad_channels_file}")
            return []

        with open(bad_channels_file, 'r') as f:
            bad_channels_data = json.load(f)

        bad_channel_ids = bad_channels_data.get('bad_channel_ids', [])
        logger.info(f"Loaded {len(bad_channel_ids)} bad channels from {bad_channels_file}")
        return bad_channel_ids

    except Exception as e:
        logger.error(f"Exception loading bad channels: {e}")
        return []

def load_recordings_and_sorting(params, probe_idx, logger, shank_idx=None, shank_rec=None):
    """
    Load recordings and sorting data for a probe (or one shank of a probe).

    When shank_idx is given, folder dict keys are expected in "p_shank_s" format
    (e.g. "0_shank0") and shank_rec is the already-split sub-recording from
    split_by("group"). Bad channels are loaded from the per-shank JSON.

    Returns:
        tuple: (raw_rec, rec, sorting), all None on failure.
    """
    try:
        folder_key = f"{probe_idx}_shank{shank_idx}" if shank_idx is not None else str(probe_idx)
        tag        = f"probe {probe_idx} shank {shank_idx}" if shank_idx is not None else f"probe {probe_idx}"

        if folder_key not in params['SORTER_OUTPUT_FOLDERS']:
            logger.error(f"No sorter output folder specified for {tag} (key '{folder_key}')")
            return None, None, None

        sorter_output_folder = params['SORTER_OUTPUT_FOLDERS'][folder_key]
        if not os.path.isdir(sorter_output_folder):
            logger.error(f"Sorter output folder does not exist: {sorter_output_folder}")
            return None, None, None

        # Raw recording: use the provided shank sub-recording or read the full probe
        if shank_rec is not None:
            raw_rec = shank_rec
        else:
            logger.info(f"Loading raw recording for {tag}...")
            raw_rec = si.read_spikeglx(params['CATGT_OUTPUT_DIR'], stream_name=f"imec{probe_idx}.ap")

        bad_channel_ids = load_bad_channels(
            params['CATGT_OUTPUT_DIR'], probe_idx, logger, shank_idx=shank_idx
        )

        logger.info(f"Removing {len(bad_channel_ids)} bad channels for {tag}...")
        rec = raw_rec.remove_channels(bad_channel_ids)

        logger.info(f"Loading Kilosort sorting output for {tag}...")
        sorting = si.read_kilosort(folder_path=sorter_output_folder)

        if folder_key in params['PREPROCESS_FOLDERS']:
            preprocess_folder = params['PREPROCESS_FOLDERS'][folder_key]
            binary_file = os.path.join(preprocess_folder, "traces_cached_seg0.raw")

            if os.path.isfile(binary_file):
                logger.info(f"Loading preprocessed recording from {binary_file}...")
                this_rec = si.read_binary(
                    binary_file,
                    sampling_frequency=rec.get_sampling_frequency(),
                    dtype='int16',
                    num_channels=rec.get_num_channels()
                )

                this_rec.annotate(is_filtered=rec.is_filtered())
                for key in rec.get_property_keys():
                    this_rec.set_property(key, rec.get_property(key))
                this_rec.set_channel_gains(rec.get_channel_gains())
                this_rec.set_probe(rec.get_probe())
                if 'probes_info' in rec.get_annotation_keys():
                    this_rec.annotate(probes_info=rec.get_annotation('probes_info'))
                if 'probe_0_planar_contour' in rec.get_annotation_keys():
                    this_rec.annotate(probe_0_planar_contour=rec.get_annotation('probe_0_planar_contour'))

                if not np.isclose(rec.get_total_duration(), this_rec.get_total_duration()):
                    logger.error(f"Duration mismatch between raw and preprocessed recordings for {tag}!")
                    return None, None, None

                rec = this_rec

        sorting.register_recording(rec)
        return raw_rec, rec, sorting

    except Exception as e:
        logger.error(f"Exception loading recordings and sorting for {tag}: {e}")
        return None, None, None

def extract_units(rec, sorting, unit_type, probe_idx, params, logger):
    """
    Extract units of a specific type (good or mua) and their properties
    
    Args:
        rec: SpikeInterface recording object
        sorting: SpikeInterface sorting object
        unit_type: Type of units to extract ('good' or 'mua')
        probe_idx: Index of the probe
        params: Dictionary of parameters
        logger: Logger object
    
    Returns:
        dict: Dictionary of unit properties or None if failed
    """
    try:
        logger.info(f"Extracting {unit_type} units for probe {probe_idx}...")
        
        # Select units based on quality
        keep_idx = np.where(sorting.get_property('quality') == unit_type)[0]
        if len(keep_idx) == 0:
            logger.info(f"No {unit_type} units found for probe {probe_idx}")
            return None
        
        keep_units = sorting.unit_ids[keep_idx]
        units = sorting.select_units(keep_units)
        logger.info(f"Found {len(keep_units)} {unit_type} units for probe {probe_idx}")
        
        # ch values are 0-based indices into the recording (KS ran on the bad-channel-removed rec).
        # rec.channel_ids are integers [0, 1, 2, ...] so ch IS the index — no string matching needed.
        ch_indices = units.get_property('ch')
        units_ch = [rec.channel_ids[x] for x in ch_indices]
        keep = list(ch_indices)
        
        # Extract spike trains
        spike_train = [units.get_unit_spike_train(x)/units.get_sampling_frequency() 
                       for x in units.unit_ids]
        
        # Process unit IDs and metadata
        unit_ids = [f"imec{probe_idx}_{str(x)}" for x in keep_units]
        depths = units.get_property('depth') + 175 # EFBG - Add 175 um to convert from probe tip to first row - aligns with LFP depths
        
        # Get channel locations and calculate shank IDs
        all_locs = rec.get_channel_locations()
        shank_ids = [int(all_locs[idx][0]//250) for idx in keep]
        
        # Extract waveforms if requested
        peak_amp = []
        valley_amp = []
        waveforms = []
        
        if params['SAVE_WAVEFORMS']:
            logger.info(f"Extracting waveforms for {unit_type} units...")
            
            # Set up the waveform extraction
            job_kwargs = dict(n_jobs=params['NUM_JOBS'], chunk_duration='1s', progress_bar=True)
            si.set_global_job_kwargs(**job_kwargs)
            
            # Create a temporary folder for waveform extraction
            temp_folder = os.path.join(params['OUTPUT_DIR'], f"temp_{unit_type}_sorting_probe{probe_idx}")
            os.makedirs(temp_folder, exist_ok=True)
            
            # Create a sorting analyzer for waveform extraction
            analyzer = si.create_sorting_analyzer(
                sorting=units, 
                recording=rec, 
                folder=temp_folder, 
                format="binary_folder", 
                sparse=True, 
                overwrite=True
            )
            
            # Extract waveforms
            analyzer.compute("random_spikes")
            analyzer.compute("waveforms")
            analyzer.compute("templates")
            
            # Extract positive peak amplitudes
            analyzer.compute("spike_amplitudes", peak_sign="pos")
            amps = analyzer.get_extension('spike_amplitudes')
            amps_data = amps.get_data(outputs='by_unit')
            peak_amp = [amps_data[0][x] for x in units.unit_ids]
            
            # Extract waveforms
            wv = analyzer.get_extension(extension_name="waveforms")
            waveforms = [np.mean(wv.get_waveforms_one_unit(x), axis=0).T for x in units.unit_ids]
            
            # Extract negative peak amplitudes
            analyzer.delete_extension('spike_amplitudes')
            analyzer.compute("spike_amplitudes", peak_sign="neg")
            amps = analyzer.get_extension('spike_amplitudes')
            amps_data = amps.get_data(outputs='by_unit')
            valley_amp = [amps_data[0][x] for x in units.unit_ids]
        
        # Return all unit data
        return {
            'depths': depths,
            'unit_ids': unit_ids,
            'channel_ids': units_ch,
            'spike_train': spike_train,
            'shank_ids': shank_ids,
            'mean_waveforms': waveforms,
            'peak_amp': peak_amp,
            'valley_amp': valley_amp
        }
        
    except Exception as e:
        logger.error(f"Exception extracting {unit_type} units for probe {probe_idx}: {e}")
        return None

def save_units(units_data, output_dir, probe_idx, unit_type, logger):
    """
    Save extracted units to a MATLAB file
    
    Args:
        units_data: Dictionary of unit properties
        output_dir: Output directory
        probe_idx: Index of the probe
        unit_type: Type of units ('clean' or 'mua')
        logger: Logger object
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if units_data is None:
            logger.warning(f"No {unit_type} units to save for probe {probe_idx}")
            return False
        
        # Create output filename
        output_file = os.path.join(output_dir, f"{unit_type}_units_imec{probe_idx}.mat")
        
        # Save units to MATLAB file
        logger.info(f"Saving {unit_type} units to {output_file}")
        scio.savemat(output_file, units_data)
        
        logger.info(f"Successfully saved {len(units_data['unit_ids'])} {unit_type} units for probe {probe_idx}")
        return True
        
    except Exception as e:
        logger.error(f"Exception saving {unit_type} units for probe {probe_idx}: {e}")
        return False

def merge_unit_dicts(shank_dicts):
    """
    Merge a list of per-shank unit dicts into a single per-probe dict.
    List fields are extended; numpy arrays are concatenated.
    """
    merged = {}
    for key in shank_dicts[0].keys():
        values = [d[key] for d in shank_dicts]
        sample = values[0]
        if isinstance(sample, np.ndarray):
            merged[key] = np.concatenate(values)
        elif isinstance(sample, list):
            merged[key] = [item for sublist in values for item in sublist]
        else:
            merged[key] = values
    return merged


def process_probe(probe_idx, params, logger):
    """
    Process a single probe: extract good and MUA units
    
    Args:
        probe_idx: Index of the probe to process
        params: Dictionary of parameters
        logger: Logger object
    
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info(f"Processing probe {probe_idx}...")

    try:
        if params['PROCESSED_BY_SHANK']:
            # ── Per-shank mode: load each shank's sorting, merge into one per-probe output ──
            if not params['SHANK_INDICES']:
                logger.error("PROCESSED_BY_SHANK=True but SHANK_INDICES is not set in params file")
                return False
            logger.info(f"Per-shank mode: loading probe {probe_idx} recording for splitting...")
            full_rec   = si.read_spikeglx(params['CATGT_OUTPUT_DIR'], stream_name=f"imec{probe_idx}.ap")
            shank_recs = full_rec.split_by("group")

            good_shank_dicts = []
            mua_shank_dicts  = []

            for shank_idx in params['SHANK_INDICES']:
                if shank_idx not in shank_recs:
                    logger.warning(f"Shank {shank_idx} not found in recording groups, skipping")
                    continue

                logger.info(f"--- Loading probe {probe_idx} shank {shank_idx} ---")
                _, rec, sorting = load_recordings_and_sorting(
                    params, probe_idx, logger,
                    shank_idx=shank_idx, shank_rec=shank_recs[shank_idx]
                )
                if rec is None or sorting is None:
                    logger.error(f"Failed to load data for probe {probe_idx} shank {shank_idx}, skipping")
                    continue

                if params['EXTRACT_GOOD_UNITS']:
                    good = extract_units(rec, sorting, 'good', probe_idx, params, logger)
                    if good is not None:
                        good_shank_dicts.append(good)

                if params['EXTRACT_MUA_UNITS']:
                    mua = extract_units(rec, sorting, 'mua', probe_idx, params, logger)
                    if mua is not None:
                        mua_shank_dicts.append(mua)

            all_successful = True

            if params['EXTRACT_GOOD_UNITS']:
                if good_shank_dicts:
                    merged = merge_unit_dicts(good_shank_dicts)
                    if not save_units(merged, params['OUTPUT_DIR'], probe_idx, 'clean', logger):
                        all_successful = False
                else:
                    logger.info(f"No good units found across any shank for probe {probe_idx}")

            if params['EXTRACT_MUA_UNITS']:
                if mua_shank_dicts:
                    merged = merge_unit_dicts(mua_shank_dicts)
                    if not save_units(merged, params['OUTPUT_DIR'], probe_idx, 'mua', logger):
                        all_successful = False
                else:
                    logger.info(f"No MUA units found across any shank for probe {probe_idx}")

            return all_successful

        else:
            # ── Legacy per-probe mode ─────────────────────────────────────────────────────
            raw_rec, rec, sorting = load_recordings_and_sorting(params, probe_idx, logger)
            if rec is None or sorting is None:
                logger.error(f"Failed to load recordings and sorting for probe {probe_idx}")
                return False

            all_successful = True

            if params['EXTRACT_GOOD_UNITS']:
                good_units = extract_units(rec, sorting, 'good', probe_idx, params, logger)
                if good_units is not None:
                    if not save_units(good_units, params['OUTPUT_DIR'], probe_idx, 'clean', logger):
                        all_successful = False

            if params['EXTRACT_MUA_UNITS']:
                mua_units = extract_units(rec, sorting, 'mua', probe_idx, params, logger)
                if mua_units is not None:
                    if not save_units(mua_units, params['OUTPUT_DIR'], probe_idx, 'mua', logger):
                        all_successful = False

            return all_successful

    except Exception as e:
        logger.error(f"Exception processing probe {probe_idx}: {e}")
        return False

def process_session(params_file):
    """Process a single session with the given parameters"""
    # Parse parameters
    params = parse_params_file(params_file)
    
    # Setup logging
    logger = setup_logging(params['OUTPUT_DIR'], params['SESSION_ID'])
    
    logger.info(f"Processing session: {params['SESSION_ID']}")
    logger.info(f"Parameters file: {params_file}")
    logger.info(f"CatGT output directory: {params['CATGT_OUTPUT_DIR']}")
    logger.info(f"Output directory: {params['OUTPUT_DIR']}")
    logger.info(f"Processing probes: {params['PROBE_INDICES']}")
    
    # Make sure output directory exists
    os.makedirs(params['OUTPUT_DIR'], exist_ok=True)
    
    # Process each probe in the list
    all_success = True
    for probe_idx in params['PROBE_INDICES']:
        success = process_probe(probe_idx, params, logger)
        if not success:
            all_success = False
            logger.warning(f"Processing failed or was skipped for probe {probe_idx}")
        else:
            logger.info(f"Processing completed successfully for probe {probe_idx}")
    
    if all_success:
        logger.info(f"All probes processed successfully for session {params['SESSION_ID']}")
        print(f"\nAll probes processed successfully for session {params['SESSION_ID']}!")
    else:
        logger.warning(f"Processing completed with warnings or errors for some probes in session {params['SESSION_ID']}")
        print(f"\nProcessing completed with warnings or errors for some probes in session {params['SESSION_ID']}. Check the log file for details.")
    
    return all_success

def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extract units after Phy curation')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--file_params', type=str, help='Path to parameters file for a single session')
    group.add_argument('--batch_params', type=str, help='Path to a file listing multiple parameter files')
    args = parser.parse_args()
    
    if args.file_params:
        # Process a single session
        process_session(args.file_params)
    else:
        # Process multiple sessions from a batch file
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
