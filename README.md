# DED-NET

## Abstract

Low-light image enhancement (LLIE) seeks to restore degradation from noise and lighting effects and methods based on Retinex theory and transformer architectures are commonly employed to address these challenges. These methods rely on original image features, often leading to artifacts and suboptimal results. To address this issue, we propose DED-Net, which utilizes multi-scale features of the original image to first estimate the illumination information to light up the low-light image, and then restore the corruption to produce the enhanced image. Specifically, the framework first exploits an Image Feature Enhancement (IFE) Module to extract richer feature granularity based on the original image sampled at different resolutions, performing preliminary enhancement on the original image to guide the image restoration. Additionally, we use a Retinex-based method to decouple the original and enhanced images into illumination feature and illumination map, fusing the two illumination feature and the illumination map to obtain reconstructed and guidance features. Then, we design a Illumination-Guided Restoration (IGR) Module, which utilizes illumination representations to direct the modeling of non-local interactions of regions with different lighting conditions, and we also integrate depthwise separable convolutions, achieving low computation but preserving a 1 high restoration quality. Experimental evaluations on benchmark datasets (e.g., LOLv1 and LOLv2) show that DED-Net outperforms existing methods in both efficiency and restoration quality. 

## Method

• We propose DED-Net, a novel framework for low-light image enhancement that utilizes multi-scale features to estimate illumination information and restore image corruption. 

• We design an Image Feature Enhancement (IFE) Module, which extracts richer feature granularity by sampling the original image at different resolutions, guiding the image restoration process through preliminary enhancement. 



• We develop an Illumination-Guided Restoration (IGR) Module, which uses illumination representations to model non-local interactions across regions with varying lighting conditions. Additionally, we integrate depthwise separable convolutions, achieving low computational cost while maintaining high restoration quality.



 • Extensive experiments on benchmark datasets LOLv1 and LOLv2 demonstrate that DED-Net outperforms  existing methods in both efficiency and restoration quality.



## platform

My platform is like this: 

* Ubuntu 18.04.6 LTS

* NVIDIA GeForce RTX 2080 Ti

* cuda 12.2

* pytorch=1.11

* cudatoolkit=11.3

* python=3.7

  

## create environment

- Make Conda Environment

```
conda create -n DED python=3.7
conda activate DED
```



- Install Dependencies

```
conda install pytorch=1.11 torchvision cudatoolkit=11.3 -c pytorch

pip install -r requirements.txt
```



- Install BasicSR

```
python setup.py develop --no_cuda_ext
```



## prepare dataset

Download the following datasets:

LOL-v1 [Baidu Disk](https://pan.baidu.com/s/1ZAC9TWR-YeuLIkWs3L7z4g?pwd=cyh2) , [Google Drive](https://drive.google.com/file/d/1L-kqSQyrmMueBh_ziWoPFhfsAh50h20H/view?usp=sharing)

LOL-v2 [Baidu Disk](https://pan.baidu.com/s/1X4HykuVL_1WyB3LWJJhBQg?pwd=cyh2) , [Google Drive](https://drive.google.com/file/d/1Ou9EljYZW8o5dbDCf9R34FS8Pd8kEp2U/view?usp=sharing)

Exdark [Google Drive](https://drive.google.com/file/d/1BHmPgu8EsHoFDDkMGLVoXIlCth2dW6Yx/view?usp=sharing)

**Then organize these datasets as follows:**

```
    |--data   
    |    |--LOLv1
    |    |    |--Train
    |    |    |    |--input
    |    |    |    |    |--100.png
    |    |    |    |    |--101.png
    |    |    |    |     ...
    |    |    |    |--target
    |    |    |    |    |--100.png
    |    |    |    |    |--101.png
    |    |    |    |     ...
    |    |    |--Test
    |    |    |    |--input
    |    |    |    |    |--111.png
    |    |    |    |    |--146.png
    |    |    |    |     ...
    |    |    |    |--target
    |    |    |    |    |--111.png
    |    |    |    |    |--146.png
    |    |    |    |     ...
```

## Training

```
# activate the enviroment
conda activate DED
```

```
# LOL-v1
python3 basicsr/train.py --opt Options/lol1.yml
```

```
# LOL-v2_syn
python3 basicsr/train.py --opt Options/lol2s.yml
```

## Test

Download our models from : [best_psnr_lolv1.pth](https://pan.baidu.com/s/1_GZOR7GH-pgWvD7ZtxMt6w?pwd=f6ca) code: f6ca: [best_psnr_lolv2s.pth](https://pan.baidu.com/s/16wFG_T_AN55vbn3nYHQ87w?pwd=ay5e) code: ay5e. Put them in folder `pretrained_weights`.

```
# activate the environment
conda activate DED

# LOL-v1
python3 Enhancement/test_from_dataset.py --opt Options/lol1.yml --weights pretrained_weights/best_psnr_lolv1.pth --dataset LOL_v1

# LOL-v2-synthetic
python3 Enhancement/test_from_dataset.py --opt Options/lol2s.yml --weights pretrained_weights/best_psnr_lolv2s.pth --dataset LOL_v2_synthetic
```

## Verify on the Exdark dataset

Please refer to the following: [https://github.com/ultralytics/yolov3].

## Temporary Citation Formats

Haosen Dong,Shihao Cheng,Zhigang Tu,Jin Chen,Tianyou Fang.

DED-Net: A Multi-Scale Feature Enhancement Framework for Low-Light Image Enhancement.

Manuscript submitted to The Visual Computer.

Current Status:Under Review
