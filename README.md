# VCVS
This repo contains the Pytorch implementation of the nidc2021 paper - Video Summarization based on Fusing Features and Shot Segmentation  
We refer to the code of `{Supervised Video Summarization Via Multiple Feature Sets with Parallel Attention}`，proposed a shot feature extraction algorithm based on netVLAD and multiple features of the video are adaptively fused using capsule network.  
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
## DataSet
For the convenience of Chinese researchers, we upload the dataset to Baidu Netdisk：
```
Address：https://pan.baidu.com/s/1PRAn3-3QE1kzD6YHnA2Zcg 
Extraction code：uzcp 
```
Please put the downloaded data folder in the root directory of the code.
## Evaluation
To evaluate all splits in ./splits with corresponding trained models in ./src/models run the following:
```
python train.py -params parameters.json
```
## Acknowledgement
We would like to thank to JA Ghauri et al. and JA Ghauri et al. for making the preprocessed datasets publicly available.
