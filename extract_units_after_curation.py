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
    'SORTER_OUTPUT_FOLDERS': {}, # Dict mapping probe indices to Kilosort/Phy output folders
    'PREPROCESS_FOLDERS': {},  # Dict mapping probe indices to SI preprocess folders
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
                    if key in ['PROBE_INDICES']:
                        # Parse list of integers
                        params[key] = [int(x.strip()) for x in value.split(',')]
                    elif key in ['SAVE_WAVEFORMS', 'EXTRACT_GOOD_UNITS', 'EXTRACT_MUA_UNITS']:
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

def load_bad_channels(catgt_output_dir, probe_idx, logger):
    """
    Load bad channel IDs from a JSON file
    
    Args:
        catgt_output_dir: Path to the CatGT output directory
        probe_idx: Index of the probe
        logger: Logger object
    
    Returns:
        list: Bad channel IDs or empty list if file not found
    """
    try:
        # Look for the bad channels JSON file
        bad_channels_file = os.path.join(catgt_output_dir, f"imec{probe_idx}_bad_channels.json")
        
        if not os.path.isfile(bad_channels_file):
            logger.warning(f"Bad channels file not found for probe {probe_idx}: {bad_channels_file}")
            return []
        
        # Load the bad channels
        with open(bad_channels_file, 'r') as f:
            bad_channels_data = json.load(f)
        
        bad_channel_ids = bad_channels_data.get('bad_channel_ids', [])
        logger.info(f"Loaded {len(bad_channel_ids)} bad channels for probe {probe_idx}")
        return bad_channel_ids
        
    except Exception as e:
        logger.error(f"Exception loading bad channels for probe {probe_idx}: {e}")
        return []

def load_recordings_and_sorting(params, probe_idx, logger):
    """
    Load recordings and sorting data for a specific probe
    
    Args:
        params: Dictionary of parameters
        probe_idx: Index of the probe to process
        logger: Logger object
    
    Returns:
        tuple: (raw_rec, rec, sorting) where raw_rec is the unfiltered recording,
               rec is the filtered recording, and sorting is the Kilosort output
    """
    try:
        # Check if sorter output folder is provided
        if str(probe_idx) not in params['SORTER_OUTPUT_FOLDERS']:
            logger.error(f"No sorter output folder specified for probe {probe_idx}")
            return None, None, None
        
        sorter_output_folder = params['SORTER_OUTPUT_FOLDERS'][str(probe_idx)]
        
        # Check if sorter output folder exists
        if not os.path.isdir(sorter_output_folder):
            logger.error(f"Sorter output folder does not exist: {sorter_output_folder}")
            return None, None, None
        
        # Load the raw recording (catgt output)
        logger.info(f"Loading raw recording for probe {probe_idx}...")
        raw_rec = si.read_spikeglx(params['CATGT_OUTPUT_DIR'], stream_name=f"imec{probe_idx}.ap")
        
        # Load bad channels
        bad_channel_ids = load_bad_channels(params['CATGT_OUTPUT_DIR'], probe_idx, logger)
        
        # Filter and remove bad channels
        logger.info(f"Removing bad channels...")
        rec = raw_rec.remove_channels(bad_channel_ids)
        
        # Load the Kilosort sorting output
        logger.info(f"Loading Kilosort sorting output...")
        sorting = si.KiloSortSortingExtractor(folder_path=sorter_output_folder)
        
        # Register the recording with the sorting
        if str(probe_idx) in params['PREPROCESS_FOLDERS']:
            # If a preprocessed recording binary is specified, use that
            preprocess_folder = params['PREPROCESS_FOLDERS'][str(probe_idx)]
            binary_file = os.path.join(preprocess_folder, "traces_cached_seg0.raw")
            
            if os.path.isfile(binary_file):
                logger.info(f"Loading preprocessed recording from {binary_file}...")
                this_rec = si.read_binary(
                    binary_file, 
                    sampling_frequency=rec.get_sampling_frequency(),
                    dtype='int16', 
                    num_channels=rec.get_num_channels()
                )
                
                # Copy properties and annotations
                this_rec.annotate(is_filtered=rec.is_filtered())
                for key in rec.get_property_keys():
                    this_rec.set_property(key, rec.get_property(key))
                this_rec.set_channel_gains(rec.get_channel_gains())
                this_rec.set_probe(rec.get_probe())
                if 'probes_info' in rec.get_annotation_keys():
                    this_rec.annotate(probes_info=rec.get_annotation('probes_info'))
                if 'probe_0_planar_contour' in rec.get_annotation_keys():
                    this_rec.annotate(probe_0_planar_contour=rec.get_annotation('probe_0_planar_contour'))
                
                # Verify the durations match (are close)
                if not np.isclose(rec.get_total_duration(), this_rec.get_total_duration()):
                    logger.error(f"Duration mismatch between raw and preprocessed recordings!")
                    return None, None, None
                
                rec = this_rec
        
        # Register the recording with the sorting
        sorting.register_recording(rec)
        
        return raw_rec, rec, sorting
        
    except Exception as e:
        logger.error(f"Exception loading recordings and sorting for probe {probe_idx}: {e}")
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
        
        # Extract channel information
        units_ch = [f"imec{probe_idx}.ap#AP" + str(x) for x in units.get_property('ch')]
        rec_ch = [f"imec{probe_idx}.ap#" + str(x) for x in rec.get_property('channel_names')]
        
        # Find indices of unit channels in the recording
        keep = []
        for ch in units_ch:
            match = np.where(np.array(rec_ch) == ch)[0]
            if len(match) > 0:
                keep.append(match[0])
            else:
                logger.warning(f"Channel {ch} not found in recording channels")
        
        if len(keep) != len(units_ch):
            logger.warning(f"Some channels were not found in the recording")
            keep_idx = keep_idx[:len(keep)]
            keep_units = sorting.unit_ids[keep_idx]
            units = sorting.select_units(keep_units)
            units_ch = [f"imec{probe_idx}.ap#AP" + str(x) for x in units.get_property('ch')]
        
        # Extract spike trains
        spike_train = [units.get_unit_spike_train(x)/units.get_sampling_frequency() 
                       for x in units.unit_ids]
        
        # Process unit IDs and metadata
        unit_ids = [f"imec{probe_idx}_{str(x)}" for x in keep_units]
        depths = units.get_property('depth')
        
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
        # Load recordings and sorting data
        raw_rec, rec, sorting = load_recordings_and_sorting(params, probe_idx, logger)
        if rec is None or sorting is None:
            logger.error(f"Failed to load recordings and sorting for probe {probe_idx}")
            return False
        
        all_successful = True
        
        # Extract good units if requested
        if params['EXTRACT_GOOD_UNITS']:
            logger.info(f"Extracting good units for probe {probe_idx}...")
            good_units = extract_units(rec, sorting, 'good', probe_idx, params, logger)
            
            # Save good units
            if good_units is not None:
                success = save_units(good_units, params['OUTPUT_DIR'], probe_idx, 'clean', logger)
                if not success:
                    all_successful = False
        
        # Extract MUA units if requested
        if params['EXTRACT_MUA_UNITS']:
            logger.info(f"Extracting MUA units for probe {probe_idx}...")
            mua_units = extract_units(rec, sorting, 'mua', probe_idx, params, logger)
            
            # Save MUA units
            if mua_units is not None:
                success = save_units(mua_units, params['OUTPUT_DIR'], probe_idx, 'mua', logger)
                if not success:
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
