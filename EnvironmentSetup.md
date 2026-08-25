Follow these steps to create the conda enviroment necessary to run the pre-processing pipeline
```
conda create --name si_preprocess python=3.10
conda activate si_preprocess
python -m pip install kilosort==4.0.30
pip uninstall torch
conda install pytorch==2.5.1 pytorch-cuda=12.8 -c pytorch -c nvidia
pip install "spikeinterface[full]==0.104.8"
pip install pynvml==12.0.0 ipykernel==6.29.5 ipywidgets==8.1.8 ipympl==0.10.0
```

