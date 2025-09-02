# Neuropixels Processing Pipeline
# This script chains together all the processing steps for Neuropixels data in batches

# Configuration - Edit these paths as needed
$script_path = "C:\Users\mvdmlab\Documents\manishm\odor-pixels\pre-processing"
$catgt_config = "E:\Task2-SWR\batch_catgt.txt"
$lfp_ks4_config = "E:\Task2-SWR\batch_lfp_ks4.txt"
$tprime_config = "E:\Task2-SWR\batch_tprime.txt"

# Activate conda environment
Write-Host "Activating conda environment..."
conda activate si_preprocess

# Step 1: Run CatGT for preprocessing
Write-Host "Step 1: Running CatGT preprocessing..."
python "$script_path\run_catgt.py" --batch_params "$catgt_config"

# Step 2: Run LFP extraction and Kilosort4
Write-Host "Step 2: Running LFP extraction and Kilosort4..."
python "$script_path\extract_lfp_and_run_ks4.py" --batch_params "$lfp_ks4_config"

# Step 3: Run TPrime for synchronization
Write-Host "Step 3: Running TPrime synchronization..."
python "$script_path\run_tprime.py" --batch_params "$tprime_config"

Write-Host "Neuropixels processing pipeline finished"