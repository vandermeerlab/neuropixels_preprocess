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
    'ANALOG_CHANNELS': ['XA0', 'XA1', 'XA2', 'XA3', 'XA4', 'XA5', 'XA6', 'XA7'],
    'DIGITAL_CHANNELS': ['XD2', 'XD3'],
    'KS_OUTPUT_FOLDERS': {},   # Dict mapping probe indices to Kilosort output folders
    'CHECK_INTERVAL': 15,      # Time in seconds between checks for TPrime process
    'TIMEOUT': 300             # Maximum time to wait for TPrime to complete (seconds)
}

# Required parameters
REQUIRED_PARAMS = ['CATGT_OUTPUT_DIR', 'OUTPUT_DIR', 'SESSION_ID', 'TPRIME_PATH']

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
                    elif key in ['ALIGN_EVENTS', 'ALIGN_SPIKES', 'ALIGN_LFP']:
                        # Parse boolean
                        params[key] = value.lower() == 'true'
                    elif key in ['CHECK_INTERVAL', 'TIMEOUT']:
                        # Parse integer
                        params[key] = int(value)
                    elif key in ['ANALOG_CHANNELS', 'DIGITAL_CHANNELS']:
                        # Parse list of strings
                        if value.lower() == 'none':
                            params[key] = []
                        else:
                            params[key] = [x.strip() for x in value.split(',')]
                    elif key == 'KS_OUTPUT_FOLDERS':
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
        
        # Check if TO_PROBE is in PROBE_INDICES
        if params['TO_PROBE'] not in params['PROBE_INDICES']:
            raise ValueError(f"TO_PROBE ({params['TO_PROBE']}) must be in PROBE_INDICES ({params['PROBE_INDICES']})")
        
        # If aligning spikes, check that KS_OUTPUT_FOLDERS is provided for all from_probes
        if params['ALIGN_SPIKES']:
            from_probes = [p for p in params['PROBE_INDICES'] if p != params['TO_PROBE']]
            missing_folders = [p for p in from_probes if str(p) not in params['KS_OUTPUT_FOLDERS']]
            if missing_folders:
                raise ValueError(f"KS_OUTPUT_FOLDERS missing for probes: {missing_folders}")
        
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
        
        # Look for to_probe sync file (always with .xd_384_6_500.txt suffix)
        to_file_pattern = f"imec{to_probe}.ap.xd_384_6_500.txt"
        to_files = [f for f in file_list if f.endswith(to_file_pattern)]
        
        if not to_files:
            logger.error(f"No sync file found for to_probe={to_probe}")
            return None, None
        
        to_file = to_files[0]
        
        # Look for from_probe sync file (always with .xd_384_6_500.txt suffix)
        from_file_pattern = f"imec{from_probe}.ap.xd_384_6_500.txt"
        from_files = [f for f in file_list if f.endswith(from_file_pattern)]
        
        if not from_files:
            logger.error(f"No sync file found for from_probe={from_probe}")
            return None, None
        
        from_file = from_files[0]
        
        logger.info(f"Found to_file: {to_file}")
        logger.info(f"Found from_file: {from_file}")
        
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

def generate_align_spikes_command(params, to_file, from_file, from_probe, logger):
    """
    Generate a TPrime command to align spike times from Kilosort output
    
    Args:
        params: Dictionary of parameters
        to_file: Path to the reference probe sync file
        from_file: Path to the probe to be aligned sync file
        from_probe: Index of the probe to be aligned
        logger: Logger object
    
    Returns:
        tuple: (command, sorter_output_folder, output_files) TPrime command, sorter output folder, and expected output files
    """
    # Platform-specific settings
    is_windows = platform.system() == "Windows"
    tprime_exe = os.path.join(params['TPRIME_PATH'], "TPrime.exe" if is_windows else "runit.sh")
    
    # Get the sorter output folder for this probe
    try:
        sorter_output_folder = params['KS_OUTPUT_FOLDERS'][str(from_probe)]
    except KeyError:
        logger.error(f"No Kilosort output folder specified for probe {from_probe}")
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
    Generate a TPrime command to align LFP times
    
    Args:
        params: Dictionary of parameters
        to_file: Path to the reference probe sync file
        from_file: Path to the probe to be aligned sync file
        from_probe: Index of the probe to be aligned
        logger: Logger object
    
    Returns:
        tuple: (command, output_files) TPrime command and list of expected output files
    """
    # Platform-specific settings
    is_windows = platform.system() == "Windows"
    tprime_exe = os.path.join(params['TPRIME_PATH'], "TPrime.exe" if is_windows else "runit.sh")
    
    # Input and output paths for LFP times
    lfp_times_path = os.path.join(params['CATGT_OUTPUT_DIR'], f"imec{from_probe}_lfp_times.npy")
    adj_lfp_times_path = os.path.join(params['CATGT_OUTPUT_DIR'], f"imec{from_probe}_adj_lfp_times.npy")
    
    # Check if the input file exists
    if not os.path.isfile(lfp_times_path):
        logger.error(f"LFP times file not found: {lfp_times_path}")
        return None, None
    
    # Expected output files
    output_files = [adj_lfp_times_path]
    
    # Generate the TPrime command
    command = (
        f"{tprime_exe} -syncperiod=1.0 -tostream={to_file} -fromstream=5,{from_file} "
        f"-events=5,{lfp_times_path},{adj_lfp_times_path}"
    )
    
    return command, output_files

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

def align_probe(to_probe, from_probe, params, logger):
    """
    Align one probe to a reference probe
    
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
    
    # Find nidq sync file
    nidq_sync_file = find_nidq_sync_files(params['CATGT_OUTPUT_DIR'], logger)
    
    all_successful = True
    
    # Align events if requested
    if params['ALIGN_EVENTS']:
        logger.info(f"Aligning events for probe {from_probe}...")
        events_command, output_files = generate_align_events_command(params, to_file, nidq_sync_file, logger)
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
                logger.error(f"Events alignment failed for probe {from_probe}")
                all_successful = False
    
    # Align spike times if requested
    if params['ALIGN_SPIKES']:
        logger.info(f"Aligning spike times for probe {from_probe}...")
        spikes_command, sorter_output_folder, output_files = generate_align_spikes_command(
            params, to_file, from_file, from_probe, logger)
        
        if spikes_command is None or sorter_output_folder is None:
            logger.error("Failed to generate spike times alignment command")
            all_successful = False
        else:
            spikes_success = run_tprime_command(
                spikes_command, 
                output_files, 
                check_interval=params['CHECK_INTERVAL'],
                timeout=params['TIMEOUT'],
                logger=logger
            )
            if not spikes_success:
                logger.error(f"Spike times alignment failed for probe {from_probe}")
                all_successful = False
            else:
                # Process spike time adjustment (cleanup and conversion)
                cleanup_success = process_spike_time_adjustment(sorter_output_folder, logger)
                if not cleanup_success:
                    logger.error(f"Spike times adjustment cleanup failed for probe {from_probe}")
                    all_successful = False
    
    # Align LFP times if requested
    if params['ALIGN_LFP']:
        logger.info(f"Aligning LFP times for probe {from_probe}...")
        lfp_command, output_files = generate_align_lfp_command(params, to_file, from_file, from_probe, logger)
        
        if lfp_command is None:
            logger.error("Failed to generate LFP times alignment command")
            all_successful = False
        else:
            lfp_success = run_tprime_command(
                lfp_command, 
                output_files, 
                check_interval=params['CHECK_INTERVAL'],
                timeout=params['TIMEOUT'],
                logger=logger
            )
            if not lfp_success:
                logger.error(f"LFP times alignment failed for probe {from_probe}")
                all_successful = False
    
    return all_successful

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
    logger.info(f"Reference probe (to_probe): {params['TO_PROBE']}")
    logger.info(f"Probes to align: {[p for p in params['PROBE_INDICES'] if p != params['TO_PROBE']]}")
    
    # Make sure output directory exists
    os.makedirs(params['OUTPUT_DIR'], exist_ok=True)
    
    # Process each probe (except the reference)
    all_success = True
    for probe_idx in params['PROBE_INDICES']:
        # Skip the reference probe
        if probe_idx == params['TO_PROBE']:
            continue
        
        success = align_probe(params['TO_PROBE'], probe_idx, params, logger)
        if not success:
            all_success = False
            logger.warning(f"Alignment failed or was skipped for probe {probe_idx}")
        else:
            logger.info(f"Alignment completed successfully for probe {probe_idx}")
    
    if all_success:
        logger.info(f"All probes aligned successfully for session {params['SESSION_ID']}")
        print(f"\nAll probes aligned successfully for session {params['SESSION_ID']}!")
    else:
        logger.warning(f"Alignment completed with warnings or errors for some probes in session {params['SESSION_ID']}")
        print(f"\nAlignment completed with warnings or errors for some probes in session {params['SESSION_ID']}. Check the log file for details.")
    
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