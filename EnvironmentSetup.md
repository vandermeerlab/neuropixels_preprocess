```
conda create --name si_preprocess python=3.10
conda activate si_preprocess
python -m pip install kilosort==4.1.7
pip uninstall torch
# pytorch installationd depends on the cuda-version seen in the output of "nvcc -V" in ther terminal
# If that command is not found, you have to install toolkit
# Look here to see what version of CUDA can you install for your GPU: https://developer.nvidia.com/cuda/gpus
# Install appropriate version of CUDA depending on your OS from here: https://developer.nvidia.com/cuda/
# Follow the instructions here to know which version of pytorch to install based on your CUDA and os:
https://pytorch.org/get-started/previous-versions/
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118 # This is for Wineows, cuda-toolkit 11.8

# The next two lines can be replace with pip install "spikeinterface[full]==0.105.0" when it is released
pip install "spikeinterface[full]@ git+https://github.com"
pip install --upgrade numpy<=2.1

pip install pynvml==12.0.0 ipykernel==6.29.5 ipywidgets==8.1.8 ipympl==0.10.0
```
