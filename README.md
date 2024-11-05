Code for paper "Depth Super-Resolution via Deep Cross-Modality and Cross-Scale Guidance".

Environments
-----
argparse os datetime torch==2.0.1 torchvision==0.10.0

Preparing data
-----
The NYUv2 dataset can be download at <a>baidu.com.

Testing
-----
create './weights' in your root dir, and download pretrained models at <a>baidu.com.

running the code
python test.py

Training
-----
After downloading trainset, running the code
python train.py
