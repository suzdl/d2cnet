import argparse
import os

from utils import *
import numpy as np
import torchvision.transforms as transforms
from torchvision import utils
from torch import Tensor
from PIL import Image
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable

from models_ori.NET import *
# from models.dkn import DKN
from data.nyu_dataloader import *
from data.rgbdd_dataloader import *
# from data.middlebury_dataloader import Middlebury_dataset
from data.midd_train_dataloader import Middlebury_Train
import torch.nn as nn
import torch.nn.functional as F
import torch
# os.environ['CUDA_VISIBLE_DEVICE']="4"
from tqdm import tqdm

datasets = ['NYU']
from torchvision import utils

def mkdir(path):
    if os.path.exists(path) is False:
        os.makedirs(path)


scale = [16]


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_transform = transforms.Compose([transforms.ToTensor()])
for s in scale:
    up = nn.Upsample(scale_factor=s, mode='bicubic')
    path = 'weights/best_model_x16.pth'
    net = Net(num_feats=32, kernel_size=3, scale=s)
    net.load_state_dict(torch.load(path, map_location='cuda:0'))
    net.to(device)
    test_minmax = None
    for data_name in datasets:
        if data_name == 'Midd':
            dataset = Middlebury_Train(root_dir='dataset_dir', scale=s, transform=data_transform, train=False)
        else:
            dataset = NYU_v2_datset(root_dir='dataset_dir', scale=s, transform=data_transform, train=False)
            test_minmax = np.load('dataset_dir/test_minmax.npy')

        dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
        t = tqdm(iter(dataloader), leave=True, total=len(dataloader), desc="Testing (%s)" % data_name)

        data_num = len(dataloader)
        rmse = np.zeros(data_num)
        mae  = np.zeros(data_num)
        resdict = {}
        with torch.no_grad():
            net.eval()
            for idx, data in enumerate(t):
                guidance, lr, gt = data['guidance'].to(device), data['lr'].to(device), data['gt'].to(device)
                # utils.save_image(gt, "./test/gt.png" , nrow=1, normalize=False)
                out = net((guidance, lr))
                # out = up(lr)
                maxx = 255.0
                minn = 0.0
                if test_minmax is not None:
                    minmax = test_minmax[:,idx]
                    minmax = torch.from_numpy(minmax).to(device)
                    rmse[idx], mae[idx] = calc_rmse(gt[0,0], out[0,0], minmax=minmax)
                    maxx, minn = minmax[0], minmax[1]
                elif 'max' in data:
                    min, max = data['min'].cuda(), data['max'].cuda()
                    out_norm = (out * (max - min)) + min
                    rmse[idx], mae[idx] = midd_calc_rmse(gt[0,0], out_norm[0,0])
                else:
                    rmse[idx], mae[idx] = midd_calc_rmse(gt[0,0], out[0,0])
                    # print(data_name['name'] + "," + rmse[idx])
                out_normed = out * (maxx - minn) + minn
                resdict[str(idx)] = mae[idx]
                # mkdir(r'./output/' + str(data_name))
                # mkdir(r'./output/' + str(data_name) + '/sgt')
                # mkdir(r'./output/' + str(data_name) + '/bicubic')
                # utils.save_image(gt.clamp(min=0, max=1), r'./output/' + str(data_name) + '/gt/' + data['filename'][0]+ '_x_gt' + str(s) + '.png',
                                #  nrow=1, normalize=False)
                # mkdir(r'./output/' + str(data_name) + '/dkn')
                # utils.save_image(out.clamp(min=0, max=1), r'./output/' + str(data_name) + '/dkn/' + data['filename'][0]+ '_x' + str(s) + '.png',
                                #  nrow=1, normalize=False)
                # if(idx>=100):
                #     break
            print(data_name + " x" + str(s) + " :", rmse.mean())
            print(sorted(resdict.items(), key = lambda kv:(kv[1], kv[0])))  



