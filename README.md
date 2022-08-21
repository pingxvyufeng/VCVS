# VCVS
This repo contains the Pytorch implementation of the Video summary model. This project adds a capsule network to the encoder-decoder model with self-attention mechanism to achieve adaptive fusion of video multimodal features. The details of the model can refer to the following figure.
## Installation
The development and evaluation was done on the following configuration:
### System configuration
* Platform :Ubuntu 18.04.4 LTS
* Display driver : NVIDIA-SMI 450.51.05
* GPU: NVIDIA 2070 super
* CUDA: 9.0.176
* CUDNN: 7.1.2
### Python packages
The main requirements are python (v3.6) and pytorch (v1.6.0). Some dependencies may need to be downloaded by the user. You can run the following commands to create the necessary environment.
```
git clone git@github.com/pingxvyufeng/VCVS.git`  
cd VCVS 
conda create -n VCVS python=3.6 
conda activate VSVS  
pip install -r requirements.txt
```
