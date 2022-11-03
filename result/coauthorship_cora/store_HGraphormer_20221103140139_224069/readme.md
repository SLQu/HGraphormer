

# Creat runing virtual Conda environment
We recommend creating a virtual Conda environment to run HGraphormer.
Following is the instruction:

conda create --name HGraphormer python=3.7
conda activate HGraphormer

conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch

pip install path
pip install pynvml
pip install scipy

## this is to check the verson of PyTorch and whether GPU is used.
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.get_device_name(0))



