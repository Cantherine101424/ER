# ER 🚩
## This repository is an official implementation of the paper  "ER: Extract-Regress Network for Precise 3D Reconstruction of Interacting Hands from Monocular Images".
![](three.jpg)
### Abstract
Reconstructing two interacting hands from a monocular RGB image presents a formidable challenge due to similar appearances and mutual occlusions. To address these issues, we propose the Extract-Regress Network (ER), 
a unified framework comprising specialized modules for visual feature extraction, feature fusion, and 3D mesh vertex regression. 
The Extract module includes DeSim for decoupling and capturing appearance details of separate hands and DeOcc for processing latent connections and spatial clues from interacting hands. 
The Regress module employs FuseJoint to enhance feature representation by fusing joint position messages into visual feature maps. 
Our approach achieves state-of-the-art performance on the InterHand2.6M dataset, with a mean per joint position error (MPJPE) of 6.65 mm, 
outperforming existing methods by significant margins. This work advances the field of image-based 3D hand reconstruction, 
offering robust solutions for virtual reality, augmented reality, and human-computer interaction applications.

#### Getting started 🥰 

- Clone this repo.
```bash
git clone https://github.com/Cantherine101424/ER
cd ER/main
```

- Install dependencies. (Python 3.8 + NVIDIA GPU + CUDA. Recommend to use Anaconda)

- Download InterHand2.6M [[HOMEPAGE](https://mks0601.github.io/InterHand2.6M/)]. 

- Download HIC [[HOMEPAGE](https://files.is.tue.mpg.de/dtzionas/Hand-Object-Capture/)] [[annotations](https://drive.google.com/file/d/1oqquzJ7DY728M8zQoCYvvuZEBh8L8zkQ/view?usp=share_link)]. You need to download 1) all `Hand-Hand Interaction` sequences (`01.zip`-`14.zip`) and 2) some of `Hand-Object Interaction` seuqneces (`15.zip`-`21.zip`) and 3) MANO fits.
#### Training
We're in the process of organizing the work and will make a full announcement when it's done.🚀🚀🚀

#### Testing
```bash
python test.py
```
