# VCVS
This repo contains the Pytorch implementation of the Video summary model. This project adds a capsule network to the encoder-decoder model with self-attention mechanism to achieve adaptive fusion of video multimodal features. The details of the model can refer to the following figure.
## Installation  
The development and evaluation was done on the following configuration:  
### System configuration
	@@ -19,16 +18,17 @@ conda activate VSVS
pip install -r requirements.txt
```
## DataSet
This model uses the TVSum and SumMe datasets commonly used in the field of video summarization for training and testing. We refer to the work of JA Ghauri et al. to extract the RGB features and Light Flow features of the video. We directly modified the H5 file of the dataset published by ZHOU et al., adding two key values, RGB_features and Light flow_features, and the model can be called directly after importing the data.At the same time, we also provide a video summary dataset we collected from the bilibili video website. This dataset is not fully opened at present, only the H5 file of the dataset is provided. For the convenience of Chinese researchers, we upload the dataset to Baidu Netdisk：
```
Address：https://pan.baidu.com/s/127tpXUrUeVpEutcokRXuoA
Extraction code：p9du 
```
Please put the downloaded data folder in the root directory of the code.
## train
You can execute the following commands to complete the training and testing of the model:
```
python main.py --train
```
You can find the training and testing results in /data/results.txt
## Acknowledgement
We would like to thank to JA Ghauri et al.  for making the preprocessed datasets publicly available.
