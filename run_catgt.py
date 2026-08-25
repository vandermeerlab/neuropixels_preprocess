#!/usr/bin/env python3
"""
Run CatGT for Neuropixels data preprocessing

This script runs CatGT to preprocess Neuropixels data, monitoring for process completion
by checking for any running CatGT processes.

Usage:
    python run_catgt.py --file_params path/to/params.txt
    python run_catgt.py --batch_params path/to/batch_list.txt
"""

import os
import sys
import time
import subprocess
import psutil
import platform
import logging
import argparse
from datetime import datetime
import re

# Default parameters
DEFAULT_PARAMS = {
    'SOURCE_DIR': None,  # Required
    'DEST_DIR': None,    # Required
    'CATGT_PATH': None,  # Required
    'SESSION_NAME': None,  # Required
    'PROBE_INDICES': None,   # None = NIDQ-only; list of ints = also process those probe AP bands
    'CHECK_INTERVAL': 120,
    'ANALOG_CHANNELS': {
        'XA0': [1, 3],
        'XA1': [1, 3],
        'XA2': [1, 3],
        'XA3': [1, 3],
        'XA4': [1, 3],
        'XA5': [1, 3],
        'XA6': [1, 3],
        'XA7': [1, 3]
    },
    'DIGITAL_CHANNELS': ['XD2', 'XD3'],
    'NUM_ANALOG_CHANNELS': 8,
    'MANUAL_PARAMS': []
}

# Required parameters
REQUIRED_PARAMS = ['SOURCE_DIR', 'DEST_DIR', 'CATGT_PATH', 'SESSION_NAME']

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
                    if key == 'PROBE_INDICES':
                        stripped = value.strip()
                        if not stripped or stripped.lower() == 'none':
                            params[key] = None
                        else:
                            params[key] = [int(x.strip()) for x in stripped.split(',')]
                    elif key == 'CHECK_INTERVAL':
                        # Parse integer
                        params[key] = int(value)
                    elif key == 'ANALOG_CHANNELS':
                        # Parse JSON
                        import json
                        params[key] = json.loads(value)
                    elif key == 'DIGITAL_CHANNELS':
                        # Parse list of strings
                        params[key] = [x.strip() for x in value.strip('[]').split(',')]
                    elif key == 'NUM_ANALOG_CHANNELS':
                        # Parse integer
                        params[key] = int(value)
                    elif key == 'MANUAL_PARAMS':
                        # Parse JSON list of strings, e.g. ["-no_auto_sync", "-no_tshift"]
                        import json
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
def setup_logging(output_dir, session_name):
    log_file = os.path.join(output_dir, f"{session_name}_catgt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
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

def generate_catgt_command(params):
    """
    Generate a CatGT command that combines preprocessing and event extraction
    
    Args:
        params: Dictionary of parameters
    
    Returns:
        str: The CatGT command
    """
    # Platform-specific settings
    is_windows = platform.system() == "Windows"
    catgt_exe = os.path.join(params['CATGT_PATH'], "CatGT.exe" if is_windows else "CatGT")
    
    # Build probe flags only when probes are requested
    probe_flags = ""
    if params['PROBE_INDICES']:
        probe_param = ','.join(map(str, params['PROBE_INDICES']))
        probe_flags = f" -ap -prb={probe_param}"

    command_prefix = f"{catgt_exe} -dir={params['SOURCE_DIR']} -run={params['SESSION_NAME']} -g=0 -t=0 -prb_fld -t_miss_ok -ni{probe_flags}"
    
    # Generate analog channel parameters
    analog_params = ""
    for key in params['ANALOG_CHANNELS']:
        analog_params += f"-xa=0,0,{key[2:]},{params['ANALOG_CHANNELS'][key][0]},{params['ANALOG_CHANNELS'][key][1]},0 "
        analog_params += f"-xia=0,0,{key[2:]},{params['ANALOG_CHANNELS'][key][0]},{params['ANALOG_CHANNELS'][key][1]},0 "
    
    # Generate digital channel parameters
    digital_params = ""
    for channel in params['DIGITAL_CHANNELS']:
        digital_params += f"-xd=0,0,{params['NUM_ANALOG_CHANNELS']},{channel[2:]},0 "
        digital_params += f"-xid=0,0,{params['NUM_ANALOG_CHANNELS']},{channel[2:]},0 "
    
    # Destination parameter
    command_suffix = f"-dest={params['DEST_DIR']}"

    # Manual extra flags
    manual_params = ' '.join(params['MANUAL_PARAMS'])

    # Build the complete command
    command = f"{command_prefix} {analog_params}{digital_params} {command_suffix}"
    if manual_params:
        command = f"{command} {manual_params}"
    
    return command

def run_catgt_command(command, check_interval, logger):
    """
    Run a CatGT command and monitor its execution
    
    Args:
        command (str): The CatGT command to run
        check_interval (int): Time in seconds between checks for CatGT process
        logger (logging.Logger): Logger object
    
    Returns:
        bool: True if the command completed successfully, False otherwise
    """
    # Check if CatGT.log exists and delete it
    if os.path.exists("CatGT.log"):
        try:
            os.remove("CatGT.log")
            logger.info("Removed existing CatGT.log file")
        except Exception as e:
            logger.warning(f"Could not remove existing CatGT.log file: {e}")
    
    start_time = time.time()
    
    try:
        # Run the command
        logger.info(f"Running command: {command}")
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        logger.info("Monitoring for CatGT processes...")
        
        # Initially wait a bit to let the process start
        time.sleep(10)
        
        # Monitor for CatGT processes
        while True:
            # Only check at the specified interval
            time.sleep(check_interval)
            
            # Check if any CatGT process is running
            catgt_running = False
            for proc in psutil.process_iter(['name']):
                try:
                    proc_name = proc.info['name'].lower()
                    if 'catgt' in proc_name:
                        catgt_running = True
                        logger.debug(f"Found running CatGT process: {proc_name}")
                        break
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
            
            if not catgt_running:
                logger.info("No CatGT processes found running")
                break
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Wait a moment for the log file to be fully written
        time.sleep(5)
        
        # Check for errors in the log file
        log_errors = check_catgt_log("CatGT.log", logger)
        if log_errors:
            logger.error("Found errors in CatGT.log")
            return False
        else:
            logger.info(f"CatGT process completed successfully in {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
            return True
            
    except Exception as e:
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.error(f"Exception running CatGT command after {elapsed_time:.1f} seconds: {e}")
        return False

def check_catgt_log(log_file, logger):
    """
    Check the CatGT log file for errors
    
    Args:
        log_file (str): Path to the CatGT log file
        logger (logging.Logger): Logger object
    
    Returns:
        bool: True if errors were found, False otherwise
    """
    try:
        if not os.path.exists(log_file):
            logger.warning(f"CatGT log file {log_file} not found")
            return False
        
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            log_content = f.read()
        
        # Look for common error patterns
        error_patterns = [
            "Error:", "ERROR:", "*** ERROR:",
            "Error creating dir", "Meta file not found"
        ]
        
        for pattern in error_patterns:
            if pattern in log_content:
                logger.error(f"Found error pattern '{pattern}' in CatGT log file")
                
                # Extract the specific error message
                error_lines = []
                for line in log_content.split('\n'):
                    if pattern in line:
                        error_lines.append(line)
                        # Also include the next few lines for context
                        for i in range(1, 3):
                            if i < len(log_content.split('\n')):
                                error_lines.append(log_content.split('\n')[-i])
                
                logger.error(f"Error details: {'; '.join(error_lines)}")
                return True
        
        return False
        
    except Exception as e:
        logger.error(f"Exception checking CatGT log file: {e}")
        return True  # Assume error if we can't check the log

def process_session(params_file):
    """Process a single session with the given parameters"""
    # Parse parameters
    params = parse_params_file(params_file)
    
    # Setup logging
    logger = setup_logging(params['DEST_DIR'], params['SESSION_NAME'])
    
    logger.info(f"Processing session: {params['SESSION_NAME']}")
    logger.info(f"Parameters file: {params_file}")
    logger.info(f"Source directory: {params['SOURCE_DIR']}")
    logger.info(f"Destination directory: {params['DEST_DIR']}")
    if params['PROBE_INDICES']:
        logger.info(f"Processing probes: {params['PROBE_INDICES']}")
    else:
        logger.info("No probes specified - NIDQ-only run")
    
    # Make sure output directory exists
    os.makedirs(params['DEST_DIR'], exist_ok=True)
    
    # Generate the command
    command = generate_catgt_command(params)
    
    # Run the command
    success = run_catgt_command(command, params['CHECK_INTERVAL'], logger)
    
    if success:
        logger.info(f"CatGT processing completed successfully for session {params['SESSION_NAME']}")
        print(f"\nCatGT processing completed successfully for session {params['SESSION_NAME']}!")
    else:
        logger.error(f"CatGT processing failed for session {params['SESSION_NAME']}")
        print(f"\nCatGT processing failed for session {params['SESSION_NAME']}. Check the log file for details.")
    
    return success

def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run CatGT for Neuropixels data preprocessing')
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