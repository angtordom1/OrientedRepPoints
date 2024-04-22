# Oriented Object Detection in High Resolution Satellite Imagery 
Object detection from aerial images is a complex task due to the difficulty in detecting small objects, the variability in their appearance, and their high density within a limited image space. To address these challenges, advanced techniques are required. Furthermore, objects in aerial images may appear with arbitrary rotations. Traditional detectors, which attempt to address the aforementioned issues, are unable to solve this orientation problem, resulting in suboptimal localization of such objects. Several proposals have been made to adapt bounding boxes to rotated objects, including the use of adaptive point sets tailored to each object's characteristics. [Oriented RepPoints](https://github.com/LiWentomng/OrientedRepPoints), is a model that uses adaptive point sets for object-oriented detection through satellite imagery. This repository is a modified version where the feature extractor is replaced by a PAFPN to improve its performance in detecting the features of different objects. Finally, a comparison is made between the base model, the modification, and other state-of-the-art models on the DOTA dataset.

The paper can be seen [here](https://oa.upm.es/75859/).

# Modified architecture
<img src="docs\overallnetwork_pafpn.png" width="800px">


# Installation
## Requirements
* Linux
* Python 3.7+ 
* PyTorch==1.4.0 or higher
* CUDA 9.0 or higher
* mmdet==1.1.0
* [mmcv](https://github.com/open-mmlab/mmcv)==0.6.2
* GCC 4.9 or higher
* NCCL 2

We have tested the following versions of OS and softwares：
* OS：Ubuntu 16.04
* CUDA: 10.0
* NCCL: 2.1.15/2.2.13/2.3.7/2.4.2
* GCC(G++): 4.9/5.3/5.4/7.3

## Install 
From [Oriented RepPoints](https://github.com/LiWentomng/OrientedRepPoints):
a. Create a conda virtual environment and activate it.  
```
conda create -n orientedreppoints python=3.7 -y 
source activate orientedreppoints
```
b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/get-started/previous-versions/), e.g.,
```
conda install pytorch=1.4.0 torchvision cudatoolkit=10.0 -c pytorch
```
c. Clone the orientedreppoints repository.
```
git clone https://github.com/LiWentomng/OrientedRepPoints.git
cd OrientedRepPoints
```
d. Install orientedreppoints.

```python 
pip install -r requirements.txt
python setup.py develop  #or "pip install -v -e ."
```

Or you can use the dockerfile from this repo.

```
docker build -t orpdet .
```
```
docker run -it --gpus all --name orpdet --shm-size 10G orpdet:latest bash
```

or if already initialized,

```
docker start -i orpdet
docker exec -it orpdet bash
```

## Install DOTA_devkit

```
sudo apt-get install swig
```
```
cd DOTA_devkit
swig -c++ -python polyiou.i
python setup.py build_ext --inplace
```

## Prepare dataset
It is recommended to symlink the dataset root to $orientedreppoints/data. If your folder structure is different, you may need to change the corresponding paths in config files.
```
orientedreppoints
|——mmdet
|——tools
|——configs
|——data
|  |——dota
|  |  |——trainval_split
|  |  |  |——images
|  |  |  |——labelTxt
|  |  |  |——trainval.json
|  |  |——test_split
|  |  |  |——images
|  |  |  |——test.json
```

# Results and Models
The results on DOTA test set are shown in the table below. More detailed results please see the paper.

  Model | Backbone | Neck  | Params (M)| mAP (%)| Inference Time (ms) 
 ----  | ----- | ------ |------| ------ | ------  
 OrientedReppoints | R-50 | FPN | 36.1 | 73.03 | 173
 OrientedReppoints | R-50 | PAFPN | 38.97 | - | -

Note: 
* The results are performed on the original DOTA images with 1024x1024 patches. 

The mAOE results on DOTA val set are shown in the table below.

 Model | Backbone | Neck | mAOE 
 ----  | ----- | ------  | -----
 OrientedReppoints | R-50 | FPN | 5.72° 
 OrientedReppoints | R-50 | PAFPN | -

 Note：Orientation error evaluation (mAOE) is calculated on the val subset(only train subset for training).