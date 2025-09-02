Follow these steps to create the conda enviroment necessary to run the pre-processing pipeline

```
conda create --name si_preprocess python=3.10
conda activate si_preprocess
python -m pip install kilosort
pip uninstall torch
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
pip install "spikeinterface[full]"
pip install pynvml ipykernel ipywidgets ipympl
```
