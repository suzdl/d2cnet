Code for paper "Depth Super-Resolution via Deep Cross-Modality and Cross-Scale Guidance".

Environments
-----
argparse  
os  
datetime  
torch==2.0.1  
torchvision==0.10.0

Preparing data
-----
The NYUv2 dataset can be downloaded at [here](https://drive.google.com/drive/folders/1SoB1Zh4c4RQEdAWL37BbY57kh0bCz2Rp?dmr=1&ec=wgc-drive-globalnav-goto).

Testing
-----
create './weights' in your root dir, and download pre-trained models at [here](https://drive.google.com/drive/folders/1ZfAglfrddQrkUEmsYEaOU9DAAIpGs-MH?dmr=1&ec=wgc-drive-globalnav-goto).

running the code  
`python test.py`

Training
-----
After downloading the trainset, running the code  
`python train.py`
