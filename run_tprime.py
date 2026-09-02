#!/usr/bin/env python3
"""
Run TPrime for Neuropixels data synchronization

This script generates and runs TPrime commands to align events, spike times, and LFP times
across different probes in Neuropixels recordings. It supports aligning:
1) Events (analog and digital channel data)
2) Spike times from Kilosort output
3) LFP time vectors

Usage:
    python run_tprime.py --file_params path/to/params.txt
    python run_tprime.py --batch_params path/to/batch_list.txt
"""

import os
import re
import sys
import json
import time
import shutil
import logging
import argparse
import platform
import numpy as np
import subprocess
from datetime import datetime

# Default parameters
DEFAULT_PARAMS = {
    'CATGT_OUTPUT_DIR': None,  # Required
    'OUTPUT_DIR': None,        # Required
    'SESSION_ID': None,        # Required
    'TPRIME_PATH': None,       # Required
    'PROBE_INDICES': [0, 1],   # List of all probes in the recording
    'TO_PROBE': 0,             # Reference probe for synchronization
    'ALIGN_EVENTS': True,      # Whether to align events
    'ALIGN_SPIKES': False,     # Whether to align spike times
    'ALIGN_LFP': False,        # Whether to align LFP times
    'ALIGN_RAW_ANALOG': False, # Whether to align raw analog channel time vectors
    'ANALOG_CHANNELS': ['XA0', 'XA1', 'XA2', 'XA3', 'XA4', 'XA5', 'XA6', 'XA7'],
    'DIGITAL_CHANNELS': ['XD2', 'XD3'],
    'RAW_ANALOG_CHANNELS': [], # Channels whose _raw_times.npy files should be aligned
    'KS_OUTPUT_FOLDERS': {},   # Dict mapping probe/shank keys to Kilosort output folders
    'PROCESS_BY_SHANK_FOR': [],   # Probe indices whose KS was run per-shank (e.g. Quadbase)
    'SHANK_INDICES': None,        # Default shank list for every by-shank probe
    'SHANK_INDICES_BY_PROBE': {}, # Per-probe overrides; see SHANK_INDICES_IMEC{p} params
    'CHECK_INTERVAL': 15,      # Time in seconds between checks for TPrime process
    'TIMEOUT': 300             # Maximum time to wait for TPrime to complete (seconds)
}

# Required parameters
REQUIRED_PARAMS = ['CATGT_OUTPUT_DIR', 'OUTPUT_DIR', 'SESSION_ID', 'TPRIME_PATH']

def shanks_for_probe(params, probe_idx):
    """Shank indices for one probe: per-probe override if given, else the global default."""
    return params['SHANK_INDICES_BY_PROBE'].get(probe_idx, params['SHANK_INDICES']) or []


def check_folder_dict_keys(folder_dict, param_name):
    """Reject the old bare-index key style ("0", "0_shank1") with a rename hint."""
    old_style = sorted(k for k in folder_dict if re.match(r'^\d+(_shank\d+)?$', str(k)))
    if old_style:
        renamed = ', '.join(f"'{k}' -> 'imec{k}'" for k in old_style)
        raise ValueError(
            f"{param_name} uses the old bare-index key style; prefix probe keys with "
            f"'imec': {renamed}"
        )


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
                    elif key == 'TO_PROBE':
                        # Parse integer
                        params[key] = int(value)
                    elif key in ['ALIGN_EVENTS', 'ALIGN_SPIKES', 'ALIGN_LFP', 'ALIGN_RAW_ANALOG']:
                        # Parse boolean
                        params[key] = value.lower() == 'true'
                    elif key in ['CHECK_INTERVAL', 'TIMEOUT']:
                        # Parse integer
                        params[key] = int(value)
                    elif key in ['ANALOG_CHANNELS', 'DIGITAL_CHANNELS', 'RAW_ANALOG_CHANNELS']:
                        # Parse list of strings
                        if value.lower() == 'none':
                            params[key] = []
                        else:
                            params[key] = [x.strip() for x in value.split(',')]
                    elif key == 'KS_OUTPUT_FOLDERS':
                        params[key] = json.loads(value)
                    elif key == 'PROCESS_BY_SHANK_FOR':
                        if value.lower() == 'none' or not value.strip():
                            params[key] = []
                        else:
                            params[key] = [int(x.strip()) for x in value.split(',')]
                    elif key == 'SHANK_INDICES':
                        if value.lower() == 'none':
                            params[key] = None
                        else:
                            params[key] = [int(x.strip()) for x in value.split(',')]
                    else:
                        # String parameters
                        params[key] = value
                else:
                    # Per-probe shank list overrides, e.g. SHANK_INDICES_IMEC0 = 0,1,2,3
                    match = re.match(r'^SHANK_INDICES_IMEC(\d+)$', key)
                    if match:
                        params['SHANK_INDICES_BY_PROBE'][int(match.group(1))] = [
                            int(x.strip()) for x in value.split(',')
                        ]
                    else:
                        print(f"Warning: Unknown parameter '{key}' in params file")
            except Exception as e:
                print(f"Error parsing line '{line}': {e}")
        
        # Check required parameters
        missing_params = [param for param in REQUIRED_PARAMS if params[param] is None]
        if missing_params:
            raise ValueError(f"Missing required parameters: {', '.join(missing_params)}")
        
        # Check if TO_PROBE is in PROBE_INDICES
        if params['TO_PROBE'] not in params['PROBE_INDICES']:
            raise ValueError(f"TO_PROBE ({params['TO_PROBE']}) must be in PROBE_INDICES ({params['PROBE_INDICES']})")

        check_folder_dict_keys(params['KS_OUTPUT_FOLDERS'], 'KS_OUTPUT_FOLDERS')

        # If aligning spikes, check that KS_OUTPUT_FOLDERS is provided for all from_probes/shanks.
        # Each probe is checked against its own mode, so by-shank and whole probes can be mixed.
        if params['ALIGN_SPIKES']:
            from_probes = [p for p in params['PROBE_INDICES'] if p != params['TO_PROBE']]
            missing_folders = []
            for p in from_probes:
                if p in params['PROCESS_BY_SHANK_FOR']:
                    shanks = shanks_for_probe(params, p)
                    if not shanks:
                        raise ValueError(
                            f"Probe {p} is in PROCESS_BY_SHANK_FOR but no shanks are set; "
                            f"set SHANK_INDICES or SHANK_INDICES_IMEC{p}"
                        )
                    missing_folders += [
                        f"imec{p}_shank{s}" for s in shanks
                        if f"imec{p}_shank{s}" not in params['KS_OUTPUT_FOLDERS']
                    ]
                elif f"imec{p}" not in params['KS_OUTPUT_FOLDERS']:
                    missing_folders.append(f"imec{p}")
            if missing_folders:
                raise ValueError(f"KS_OUTPUT_FOLDERS missing entries: {missing_folders}")
        
        return params
        
    except Exception as e:
        print(f"Error parsing parameters file '{params_file}': {e}")
        sys.exit(1)

# Set up logging
def setup_logging(output_dir, session_id):
    log_file = os.path.join(output_dir, f"{session_id}_tprime_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
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

def find_probe_sync_file(probe_idx, file_list, logger):
    """
    Find the AP sync file for a given probe, trying both standard NPX2 (384 ch)
    and Quadbase (1539 ch) filename patterns.

    Returns the matching path, or None if not found.
    """
    patterns = [
        f"imec{probe_idx}.ap.xd_384_6_500.txt",   # NPX2 standard
        f"imec{probe_idx}.ap.xd_1539_6_500.txt",  # NPX2 Quadbase
    ]
    for pat in patterns:
        matches = [f for f in file_list if f.endswith(pat)]
        if matches:
            logger.info(f"Found probe {probe_idx} sync file: {matches[0]}")
            return matches[0]
    logger.error(f"No sync file found for probe {probe_idx} (tried: {patterns})")
    return None


def find_nidq_sync_files(catgt_output_dir, logger):
    """
    Find the NIDQ sync file
    
    Args:
        catgt_output_dir: Path to the CatGT output directory
        logger: Logger object
    
    Returns:
        string: NIDQ sync file, none if not found
    """
    logger.info(f"Finding NIDQ sync files")
    
    try:
        # List all files in the directory
        file_list = []
        for root, dirs, files in os.walk(catgt_output_dir):
            for file in files:
                file_list.append(os.path.join(root, file))
        
        # Look for to_probe sync file (always with .xd_384_6_500.txt suffix)
        pattern_1 = 'tcat.nidq'
        pattern_2 = '500.txt'
        sync_file = [f for f in file_list if pattern_1 in f and f.endswith(pattern_2)]
        
        if not sync_file:
            logger.error(f"No sync_file found for NIDQ device")
            return None
        
        if len(sync_file) > 1:
            logger.error(f"Multiple sync_files found: {sync_file}")
            return None
        
        logger.info(f"Found sync_file for NIDQ device: {sync_file[0]}")
        return sync_file[0]
        
    except Exception as e:
        logger.error(f"Exception finding NIDQ sync: {e}")
        return None


def find_sync_files(catgt_output_dir, to_probe, from_probe, logger):
    """
    Find the synchronization files for the given probes
    
    Args:
        catgt_output_dir: Path to the CatGT output directory
        to_probe: Index of the reference probe
        from_probe: Index of the probe to be aligned
        logger: Logger object
    
    Returns:
        tuple: (to_file, from_file) paths to sync files, or (None, None) if not found
    """
    logger.info(f"Finding sync files for to_probe={to_probe} and from_probe={from_probe}")
    
    try:
        # List all files in the directory
        file_list = []
        for root, dirs, files in os.walk(catgt_output_dir):
            for file in files:
                file_list.append(os.path.join(root, file))
        
        to_file = find_probe_sync_file(to_probe, file_list, logger)
        if to_file is None:
            return None, None

        from_file = find_probe_sync_file(from_probe, file_list, logger)
        if from_file is None:
            return None, None

        return to_file, from_file
        
    except Exception as e:
        logger.error(f"Exception finding sync files: {e}")
        return None, None

def find_event_files(catgt_output_dir, channel_type, channel_number, logger):
    """
    Find ON and OFF event files for a specific channel
    
    Args:
        catgt_output_dir: Path to the CatGT output directory
        channel_type: 'XA' for analog or 'XD' for digital
        channel_number: Channel number
        logger: Logger object
    
    Returns:
        tuple: (on_file, off_file) paths to event files, or (None, None) if not found
    """
    try:
        # List all files in the directory
        file_list = []
        for root, dirs, files in os.walk(catgt_output_dir):
            for file in files:
                file_list.append(os.path.join(root, file))
        
        # For analog channels, look for nidq.xa_N_0.txt and nidq.xia_N_0.txt
        if channel_type == 'XA':
            on_file_pattern = f"nidq.xa_{channel_number}"
            off_file_pattern = f"nidq.xia_{channel_number}"
            
            # Find ON event files
            on_files = [f for f in file_list if on_file_pattern in f and '0.txt' in f]
            if not on_files:
                logger.warning(f"No ON event file found for {channel_type}{channel_number}")
                on_file = None
            else:
                on_file = on_files[0]
                
            # Find OFF event files
            off_files = [f for f in file_list if off_file_pattern in f and '0.txt' in f]
            if not off_files:
                logger.warning(f"No OFF event file found for {channel_type}{channel_number}")
                off_file = None
            else:
                off_file = off_files[0]
                
        # For digital channels, look for nidq.xd files with N_0.txt pattern
        else:  # channel_type == 'XD'
            # The pattern is different for digital channels
            on_file_pattern = f"nidq.xd"
            off_file_pattern = f"nidq.xid"
            
            # Find ON event files
            on_files = [f for f in file_list if on_file_pattern in f and f'{channel_number}_0.txt' in f]
            if not on_files:
                logger.warning(f"No ON event file found for {channel_type}{channel_number}")
                on_file = None
            else:
                on_file = on_files[0]
                
            # Find OFF event files
            off_files = [f for f in file_list if off_file_pattern in f and f'{channel_number}_0.txt' in f]
            if not off_files:
                logger.warning(f"No OFF event file found for {channel_type}{channel_number}")
                off_file = None
            else:
                off_file = off_files[0]
        
        return on_file, off_file
        
    except Exception as e:
        logger.error(f"Exception finding event files: {e}")
        return None, None

def generate_align_events_command(params, to_file, from_file, logger):
    """
    Generate a TPrime command to align events from analog and digital channels
    
    Args:
        params: Dictionary of parameters
        to_file: Path to the reference probe sync file
        from_file: Path to the probe to be aligned sync file
        logger: Logger object
    
    Returns:
        tuple: (command, output_files) TPrime command and list of expected output files
    """
    # Platform-specific settings
    is_windows = platform.system() == "Windows"
    tprime_exe = os.path.join(params['TPRIME_PATH'], "TPrime.exe" if is_windows else "runit.sh")
    
    # Base command with sync period and stream files
    command = f"{tprime_exe} -syncperiod=1.0 -tostream={to_file} -fromstream=5,{from_file}"
    
    # Track expected output files
    output_files = []
    
    # Generate events parameters for each analog channel
    events_params = ""
    for channel in params['ANALOG_CHANNELS']:
        channel_number = channel[2:]  # Extract the number from XA0, XA1, etc.
        on_file, off_file = find_event_files(params['CATGT_OUTPUT_DIR'], 'XA', channel_number, logger)
        
        if on_file:
            on_out_file = os.path.join(params['OUTPUT_DIR'], f"{channel}_ON.txt")
            events_params += f" -events=5,{on_file},{on_out_file}"
            output_files.append(on_out_file)
            
        if off_file:
            off_out_file = os.path.join(params['OUTPUT_DIR'], f"{channel}_OFF.txt")
            events_params += f" -events=5,{off_file},{off_out_file}"
            output_files.append(off_out_file)
    
    # Generate events parameters for each digital channel
    for channel in params['DIGITAL_CHANNELS']:
        channel_number = channel[2:]  # Extract the number from XD0, XD1, etc.
        on_file, off_file = find_event_files(params['CATGT_OUTPUT_DIR'], 'XD', channel_number, logger)
        
        if on_file:
            on_out_file = os.path.join(params['OUTPUT_DIR'], f"{channel}_ON.txt")
            events_params += f" -events=5,{on_file},{on_out_file}"
            output_files.append(on_out_file)
            
        if off_file:
            off_out_file = os.path.join(params['OUTPUT_DIR'], f"{channel}_OFF.txt")
            events_params += f" -events=5,{off_file},{off_out_file}"
            output_files.append(off_out_file)
    
    # Combine the base command with events parameters
    full_command = f"{command}{events_params}"
    
    return full_command, output_files

def generate_align_spikes_command(params, to_file, from_file, from_probe, logger, folder_key=None):
    """
    Generate a TPrime command to align spike times from Kilosort output.

    folder_key: key into KS_OUTPUT_FOLDERS (e.g. "imec0" for a whole probe,
                "imec0_shank2" for shank mode). Defaults to "imec{from_probe}".
    """
    is_windows = platform.system() == "Windows"
    tprime_exe = os.path.join(params['TPRIME_PATH'], "TPrime.exe" if is_windows else "runit.sh")

    if folder_key is None:
        folder_key = f"imec{from_probe}"

    try:
        sorter_output_folder = params['KS_OUTPUT_FOLDERS'][folder_key]
    except KeyError:
        logger.error(f"No Kilosort output folder for key '{folder_key}' in KS_OUTPUT_FOLDERS")
        return None, None, None
    
    # Make sure the sorter_output_folder exists
    if not os.path.isdir(sorter_output_folder):
        logger.error(f"Sorter output folder does not exist: {sorter_output_folder}")
        return None, None, None
    
    # Create spike_timestamps.npy from spike_times.npy
    spike_times_path = os.path.join(sorter_output_folder, "spike_times.npy")
    spike_timestamps_path = os.path.join(sorter_output_folder, "spike_timestamps.npy")
    
    if not os.path.isfile(spike_times_path):
        logger.error(f"spike_times.npy not found in {sorter_output_folder}")
        return None, None, None
    
    try:
        # Load spike times and convert to timestamps
        # For this we need the sampling rate, but we'll use a default of 30000 Hz if not available
        sampling_rate = 30000.0  # Default for Neuropixels
        logger.info(f"Converting spike times to timestamps using sampling rate {sampling_rate} Hz")
        spike_times = np.load(spike_times_path)
        spike_timestamps = spike_times * sampling_rate
        np.save(spike_timestamps_path, spike_timestamps)
        logger.info(f"Created {spike_timestamps_path}")
    except Exception as e:
        logger.error(f"Error creating spike_timestamps.npy: {e}")
        return None, None, None
    
    # Output path for adjusted timestamps
    adj_spike_timestamps_path = os.path.join(sorter_output_folder, "adj_spike_timestamps.npy")
    
    # Expected output files
    output_files = [adj_spike_timestamps_path]
    
    # Generate the TPrime command
    command = (
        f"{tprime_exe} -syncperiod=1.0 -tostream={to_file} -fromstream=5,{from_file} "
        f"-events=5,{spike_timestamps_path},{adj_spike_timestamps_path}"
    )
    
    return command, sorter_output_folder, output_files

def generate_align_lfp_command(params, to_file, from_file, from_probe, logger):
    """
    Generate a TPrime command to align LFP times.

    Always one file per probe (imec{p}_lfp_times.npy), including for probes sorted
    per-shank: a probe's shanks are clock-locked, so they share a single time vector.
    """
    is_windows = platform.system() == "Windows"
    tprime_exe = os.path.join(params['TPRIME_PATH'], "TPrime.exe" if is_windows else "runit.sh")

    command_base = f"{tprime_exe} -syncperiod=1.0 -tostream={to_file} -fromstream=5,{from_file}"

    lfp_times_path = os.path.join(params['CATGT_OUTPUT_DIR'], f"imec{from_probe}_lfp_times.npy")
    adj_lfp_times_path = os.path.join(params['CATGT_OUTPUT_DIR'], f"imec{from_probe}_adj_lfp_times.npy")
    if not os.path.isfile(lfp_times_path):
        logger.error(f"LFP times file not found: {lfp_times_path}")
        return None, None

    events_params = f" -events=5,{lfp_times_path},{adj_lfp_times_path}"
    output_files = [adj_lfp_times_path]

    return f"{command_base}{events_params}", output_files

def generate_align_raw_analog_command(params, to_file, nidq_sync_file, logger):
    """
    Generate a TPrime command to align raw analog channel time vectors.

    All requested channels are batched into a single TPrime call, each as a
    separate -events argument, using the NIDQ sync file as the fromstream
    (same approach as event alignment).

    Args:
        params: Dictionary of parameters
        to_file: Path to the reference probe sync file
        nidq_sync_file: Path to the NIDQ sync file
        logger: Logger object

    Returns:
        tuple: (command, output_files) or (None, None) on error
    """
    if not params['RAW_ANALOG_CHANNELS']:
        logger.error("ALIGN_RAW_ANALOG is True but RAW_ANALOG_CHANNELS is empty")
        return None, None

    if nidq_sync_file is None:
        logger.error("Cannot align raw analog channels: NIDQ sync file not found")
        return None, None

    is_windows = platform.system() == "Windows"
    tprime_exe = os.path.join(params['TPRIME_PATH'], "TPrime.exe" if is_windows else "runit.sh")

    command = f"{tprime_exe} -syncperiod=1.0 -tostream={to_file} -fromstream=5,{nidq_sync_file}"

    output_files = []
    events_params = ""

    for channel in params['RAW_ANALOG_CHANNELS']:
        raw_times_path = os.path.join(params['CATGT_OUTPUT_DIR'], f"{channel}_raw_times.npy")
        adj_raw_times_path = os.path.join(params['CATGT_OUTPUT_DIR'], f"{channel}_adj_raw_times.npy")

        if not os.path.isfile(raw_times_path):
            logger.error(f"Raw times file not found for channel {channel}: {raw_times_path}")
            return None, None

        events_params += f" -events=5,{raw_times_path},{adj_raw_times_path}"
        output_files.append(adj_raw_times_path)
        logger.info(f"  {channel}: {raw_times_path} -> {adj_raw_times_path}")

    full_command = f"{command}{events_params}"
    return full_command, output_files


def process_spike_time_adjustment(sorter_output_folder, logger):
    """
    Process spike time adjustment after TPrime is done
    
    Args:
        sorter_output_folder: Path to the Kilosort output folder
        logger: Logger object
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Input and output paths
        spike_times_path = os.path.join(sorter_output_folder, "spike_times.npy")
        old_spike_times_path = os.path.join(sorter_output_folder, "old_spike_times.npy")
        adj_spike_timestamps_path = os.path.join(sorter_output_folder, "adj_spike_timestamps.npy")
        spike_timestamps_path = os.path.join(sorter_output_folder, "spike_timestamps.npy")
        new_spike_times_path = os.path.join(sorter_output_folder, "new_spike_times.npy")
        
        # Check if adjusted timestamps file exists
        if not os.path.isfile(adj_spike_timestamps_path):
            logger.error(f"Adjusted timestamps file not found: {adj_spike_timestamps_path}")
            return False
        
        # Convert adjusted timestamps back to spike times (samples)
        try:
            # For this we need the sampling rate, but we'll use a default of 30000 Hz if not available
            sampling_rate = 30000.0  # Default for Neuropixels
            logger.info(f"Converting adjusted timestamps to spike times using sampling rate {sampling_rate} Hz")
            adj_timestamps = np.load(adj_spike_timestamps_path)
            new_spike_times = adj_timestamps * (1.0 / sampling_rate)
            np.save(new_spike_times_path, new_spike_times)
            logger.info(f"Created new spike times file: {new_spike_times_path}")
        except Exception as e:
            logger.error(f"Error converting adjusted timestamps to spike times: {e}")
            return False
        
        # Perform cleanup
        logger.info("Performing cleanup...")
        
        # Rename original spike_times.npy to old_spike_times.npy
        if os.path.isfile(spike_times_path):
            shutil.move(spike_times_path, old_spike_times_path)
            logger.info(f"Renamed {spike_times_path} to {old_spike_times_path}")
        
        # Rename new_spike_times.npy to spike_times.npy
        if os.path.isfile(new_spike_times_path):
            shutil.move(new_spike_times_path, spike_times_path)
            logger.info(f"Renamed {new_spike_times_path} to {spike_times_path}")
        
        # Delete temporary files
        for temp_file in [spike_timestamps_path, adj_spike_timestamps_path]:
            if os.path.isfile(temp_file):
                os.remove(temp_file)
                logger.info(f"Deleted temporary file: {temp_file}")
        
        logger.info("Spike time adjustment cleanup completed")
        return True
        
    except Exception as e:
        logger.error(f"Exception during spike time adjustment cleanup: {e}")
        return False

def run_tprime_command(command, output_files, check_interval=15, timeout=300, logger=None):
    """
    Run a TPrime command and monitor its execution
    
    Args:
        command (str): The TPrime command to run
        output_files (list): List of output files that should be created by TPrime
        check_interval (int): Time in seconds between checks for TPrime completion
        timeout (int): Maximum time in seconds to wait for TPrime to complete
        logger (logging.Logger): Logger object
    
    Returns:
        bool: True if the command completed successfully, False otherwise
    """
    # Check if TPrime.log exists and delete it
    if os.path.exists("TPrime.log"):
        try:
            os.remove("TPrime.log")
            logger.info("Removed existing TPrime.log file")
        except Exception as e:
            logger.warning(f"Could not remove existing TPrime.log file: {e}")
    
    start_time = time.time()
    
    try:
        # Run the command
        logger.info(f"Running command: {command}")
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        logger.info(f"Monitoring TPrime execution (timeout: {timeout}s)...")
        
        # Initially wait a bit to let the process start
        time.sleep(5)
        
        # Command verification in log
        command_verified = False
        error_found = False
        
        # Continue until timeout
        while (time.time() - start_time) < timeout:
            # Check if the process has already terminated
            if process.poll() is not None:
                logger.info("TPrime process has terminated")
                break
            
            # Check if TPrime.log exists and has content
            if os.path.exists("TPrime.log"):
                with open("TPrime.log", 'r', encoding='utf-8', errors='ignore') as f:
                    log_content = f.read()
                
                # Check if log contains the command line (indicates successful start)
                if "Cmdline:" in log_content:
                    command_verified = True
                    logger.info("TPrime command found in log file")
                
                # Look for error patterns
                error_patterns = ["Error:", "ERROR:", "Failed", "Did not understand"]
                for pattern in error_patterns:
                    if pattern in log_content:
                        logger.error(f"Found error pattern '{pattern}' in TPrime.log")
                        error_found = True
                        break
                
                if error_found:
                    break
            
            # Check if output files exist
            all_files_exist = True
            for output_file in output_files:
                if not os.path.exists(output_file):
                    all_files_exist = False
                    break
            
            if all_files_exist:
                logger.info("All expected output files have been created")
                break
            
            # Wait before checking again
            time.sleep(check_interval)
        
        # Check for timeout
        if (time.time() - start_time) >= timeout:
            logger.error(f"TPrime execution timed out after {timeout} seconds")
            # Try to terminate the process if it's still running
            if process.poll() is None:
                process.terminate()
            return False
        
        # Check if there was an error
        if error_found:
            return False
        
        # Verify all output files exist
        all_exist = True
        missing_files = []
        for output_file in output_files:
            if not os.path.exists(output_file):
                all_exist = False
                missing_files.append(output_file)
        
        if not all_exist:
            logger.error(f"TPrime did not create expected output files: {missing_files}")
            return False
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        logger.info(f"TPrime process completed successfully in {elapsed_time:.1f} seconds")
        return True
            
    except Exception as e:
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.error(f"Exception running TPrime command after {elapsed_time:.1f} seconds: {e}")
        
        # Try to terminate the process if it's still running
        if 'process' in locals() and process.poll() is None:
            process.terminate()
            
        return False

def align_nidq_channels(to_probe, params, logger):
    """
    Align NIDQ-sourced data (events and raw analog channels) to the reference probe.

    This only requires the reference probe's sync file and the NIDQ sync file —
    no from_probe is needed. It runs once per session regardless of how many probes
    are present, including single-probe recordings.

    Args:
        to_probe: Index of the reference probe
        params: Dictionary of parameters
        logger: Logger object

    Returns:
        bool: True if all requested NIDQ alignments succeeded, False otherwise
    """
    logger.info(f"Aligning NIDQ channels to reference probe {to_probe}...")

    # Build file list once, then use the shared helper for probe sync file discovery
    file_list = []
    for root, _, files in os.walk(params['CATGT_OUTPUT_DIR']):
        for f in files:
            file_list.append(os.path.join(root, f))

    to_file = find_probe_sync_file(to_probe, file_list, logger)
    if to_file is None:
        return False

    # Find NIDQ sync file
    nidq_sync_file = find_nidq_sync_files(params['CATGT_OUTPUT_DIR'], logger)

    all_successful = True

    # Align events (analog + digital channels from NIDQ)
    if params['ALIGN_EVENTS']:
        logger.info("Aligning NIDQ events...")
        events_command, output_files = generate_align_events_command(
            params, to_file, nidq_sync_file, logger)
        if events_command is None:
            logger.error("Failed to generate events alignment command")
            all_successful = False
        else:
            events_success = run_tprime_command(
                events_command,
                output_files,
                check_interval=params['CHECK_INTERVAL'],
                timeout=params['TIMEOUT'],
                logger=logger
            )
            if not events_success:
                logger.error("Events alignment failed")
                all_successful = False

    # Align raw analog channel time vectors
    if params['ALIGN_RAW_ANALOG']:
        logger.info("Aligning raw analog channel times (using NIDQ sync)...")
        raw_analog_command, output_files = generate_align_raw_analog_command(
            params, to_file, nidq_sync_file, logger)
        if raw_analog_command is None:
            logger.error("Failed to generate raw analog alignment command")
            all_successful = False
        else:
            raw_analog_success = run_tprime_command(
                raw_analog_command,
                output_files,
                check_interval=params['CHECK_INTERVAL'],
                timeout=params['TIMEOUT'],
                logger=logger
            )
            if not raw_analog_success:
                logger.error("Raw analog channel time alignment failed")
                all_successful = False

    return all_successful


def align_probe(to_probe, from_probe, params, logger):
    """
    Align spike times and LFP from one probe to the reference probe.

    This requires sync files for both probes. NIDQ-based alignment (events,
    raw analog) is handled separately by align_nidq_channels.

    Args:
        to_probe: Index of the reference probe
        from_probe: Index of the probe to be aligned
        params: Dictionary of parameters
        logger: Logger object

    Returns:
        bool: True if all alignment operations were successful, False otherwise
    """
    logger.info(f"Aligning probe {from_probe} to reference probe {to_probe}")

    # Find sync files for the two probes
    to_file, from_file = find_sync_files(params['CATGT_OUTPUT_DIR'], to_probe, from_probe, logger)
    if to_file is None or from_file is None:
        logger.error(f"Cannot align probe {from_probe} to {to_probe}: sync files not found")
        return False

    all_successful = True
    by_shank = from_probe in params['PROCESS_BY_SHANK_FOR']
    shank_indices = shanks_for_probe(params, from_probe) if by_shank else None

    # Align spike times
    if params['ALIGN_SPIKES']:
        shank_list = shank_indices if by_shank else [None]
        for s in shank_list:
            label = f"probe {from_probe}" + (f" shank {s}" if s is not None else "")
            folder_key = f"imec{from_probe}_shank{s}" if s is not None else None
            logger.info(f"Aligning spike times for {label}...")
            spikes_command, sorter_output_folder, output_files = generate_align_spikes_command(
                params, to_file, from_file, from_probe, logger, folder_key=folder_key)
            if spikes_command is None or sorter_output_folder is None:
                logger.error(f"Failed to generate spike times alignment command for {label}")
                all_successful = False
                continue
            spikes_success = run_tprime_command(
                spikes_command, output_files,
                check_interval=params['CHECK_INTERVAL'],
                timeout=params['TIMEOUT'], logger=logger)
            if not spikes_success:
                logger.error(f"Spike times alignment failed for {label}")
                all_successful = False
            else:
                cleanup_success = process_spike_time_adjustment(sorter_output_folder, logger)
                if not cleanup_success:
                    logger.error(f"Spike times adjustment cleanup failed for {label}")
                    all_successful = False

    # Align LFP times (one time vector per probe, shanks included)
    if params['ALIGN_LFP']:
        logger.info(f"Aligning LFP times for probe {from_probe}...")
        lfp_command, output_files = generate_align_lfp_command(
            params, to_file, from_file, from_probe, logger)
        if lfp_command is None:
            logger.error(f"Failed to generate LFP times alignment command for probe {from_probe}")
            all_successful = False
        else:
            lfp_success = run_tprime_command(
                lfp_command, output_files,
                check_interval=params['CHECK_INTERVAL'],
                timeout=params['TIMEOUT'], logger=logger)
            if not lfp_success:
                logger.error(f"LFP times alignment failed for probe {from_probe}")
                all_successful = False

    return all_successful


def process_session(params_file):
    """Process a single session with the given parameters"""
    params = parse_params_file(params_file)

    logger = setup_logging(params['OUTPUT_DIR'], params['SESSION_ID'])

    logger.info(f"Processing session: {params['SESSION_ID']}")
    logger.info(f"Parameters file: {params_file}")
    logger.info(f"CatGT output directory: {params['CATGT_OUTPUT_DIR']}")
    logger.info(f"Output directory: {params['OUTPUT_DIR']}")
    logger.info(f"Reference probe (to_probe): {params['TO_PROBE']}")
    from_probes = [p for p in params['PROBE_INDICES'] if p != params['TO_PROBE']]
    logger.info(f"Probes to align: {from_probes}")
    logger.info(f"Probes sorted by shank: {params['PROCESS_BY_SHANK_FOR']}")

    os.makedirs(params['OUTPUT_DIR'], exist_ok=True)

    all_success = True

    # Step 1: NIDQ-based alignment (events + raw analog).
    # Runs once regardless of probe count — works even with a single probe.
    if params['ALIGN_EVENTS'] or params['ALIGN_RAW_ANALOG']:
        nidq_success = align_nidq_channels(params['TO_PROBE'], params, logger)
        if not nidq_success:
            all_success = False
            logger.warning("NIDQ channel alignment completed with errors")

    # Step 2: Probe-to-probe alignment (spikes + LFP) for each non-reference probe.
    for probe_idx in from_probes:
        if not (params['ALIGN_SPIKES'] or params['ALIGN_LFP']):
            logger.info(f"No spike or LFP alignment requested for probe {probe_idx}, skipping")
            continue

        success = align_probe(params['TO_PROBE'], probe_idx, params, logger)
        if not success:
            all_success = False
            logger.warning(f"Probe alignment failed or was skipped for probe {probe_idx}")
        else:
            logger.info(f"Probe alignment completed successfully for probe {probe_idx}")

    if all_success:
        logger.info(f"All alignment completed successfully for session {params['SESSION_ID']}")
        print(f"\nAll alignment completed successfully for session {params['SESSION_ID']}!")
    else:
        logger.warning(f"Alignment completed with warnings or errors for session {params['SESSION_ID']}")
        print(f"\nAlignment completed with warnings or errors for session {params['SESSION_ID']}. Check the log file for details.")

    return all_success

def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run TPrime for Neuropixels data synchronization')
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