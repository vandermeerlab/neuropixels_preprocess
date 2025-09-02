# Neuropixels Processing Pipeline
# This script chains together all the processing steps for Neuropixels data

# Configuration - Edit these paths as needed
$script_path = "C:\Users\mvdmlab\Documents\manishm\odor-pixels\pre-processing"
$catgt_config = "E:\Task2-SWR\M590\M590-2025-03-31_catgt_params.txt"
$lfp_ks4_config = "E:\Task2-SWR\M590\M590-2025-03-31_lfp_ks4_params.txt"
$tprime_config = "E:\Task2-SWR\M590\M590-2025-03-31_tprime_params.txt"

# Activate conda environment
Write-Host "Activating conda environment..."
conda activate si_preprocess

# Step 1: Run CatGT for preprocessing
Write-Host "Step 1: Running CatGT preprocessing..."
python "$script_path\run_catgt.py" --file_params "$catgt_config"

# Step 2: Run LFP extraction and Kilosort4
Write-Host "Step 2: Running LFP extraction and Kilosort4..."
python "$script_path\extract_lfp_and_run_ks4.py" --file_params "$lfp_ks4_config"

# Step 3: Run TPrime for synchronization
Write-Host "Step 3: Running TPrime synchronization..."
python "$script_path\run_tprime.py" --file_params "$tprime_config"

Write-Host "Neuropixels processing pipeline finished"