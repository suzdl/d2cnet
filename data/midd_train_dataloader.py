import torch.utils.data as data
import torch
import numpy as np
import os
import random
from PIL import Image
import torchvision.transforms as trans
from scipy import io as sio

def get_patch(img, gt, patch_size=16):
    th, tw = img.shape[:2]  ## HR image

    tp = round(patch_size)

    tx = random.randrange(0, (tw-tp))
    ty = random.randrange(0, (th-tp))
    # lr_tx = tx // scale
    # lr_ty = ty // scale

    return img[ty:ty + tp, tx:tx + tp, :], gt[ty:ty + tp, tx:tx + tp, :]

def mod_crop(img, modulo):
    h, w = img.shape[: 2]
    return img[: h - (h % modulo), :w - (w % modulo)]

def img_resize(gt_img, rgb_img, scale):
    rh, rw = rgb_img.shape[: 2]
    dh, dw = gt_img.shape[: 2]
    if rh != dh:
        crop_h = (rh - dh) // 2
        crop_w = (rw - dw) // 2
        rgb_img = rgb_img[crop_h: rh - crop_h, crop_w: rw - crop_w, :]

    gt_img, rgb_img = mod_crop(gt_img, modulo=scale), mod_crop(rgb_img, scale)
    return gt_img, rgb_img

def get_img_file(file_name):
    imagelist = []
    for parent, dirnames, filenames in os.walk(file_name):
        for filename in filenames:
            if filename.lower().endswith(
                    ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff', '.npy', '.mat')):
                imagelist.append(os.path.join(parent, filename))
        return imagelist


"training midd"


class Middlebury_Train(data.Dataset):
    def __init__(self, root_dir, scale=4, train=False, transform=None):
        super(Middlebury_Train, self).__init__()

        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.scale = scale
        self.hor_flip = trans.RandomHorizontalFlip(1)
        self.ver_flip = trans.RandomVerticalFlip(1)
        self.depths = sorted(get_img_file(root_dir + '/depth'))
        self.images = sorted(get_img_file(root_dir + '/rgb'))

    def __getitem__(self, idx):

        if self.train:
            depth = np.load(self.depths[idx])
            image = np.transpose(np.load(self.images[idx]) / 255.0, [1, 2, 0])
        else:
            image = np.array(Image.open(self.images[idx]).convert("RGB")).astype(np.float32) / 255.0
            depth = np.array(Image.open(self.depths[idx]))
        maxx = depth.max()
        minn = depth.min()
        depth = (depth - minn) / (maxx - minn)
        depth, image = img_resize(depth, image, self.scale)

        if self.train:
            image, depth = get_patch(img=image, gt=np.expand_dims(depth, 2), patch_size=256)
        h, w = depth.shape[:2]
        s = self.scale
        lr = np.array(Image.fromarray(depth.squeeze()).resize((w // s, h // s), Image.BICUBIC))

        if self.transform:
            image = self.transform(image).float()
            depth = self.transform(depth).float()
            lr = self.transform(np.expand_dims(lr, 2)).float()
        
        if self.train:
            # 数据增强
            p_ver_flip = 0.5
            p_hor_flip = 0.5
            if np.random.uniform(0.0, 1.0) < p_ver_flip:
                image = self.ver_flip(image)
                depth = self.ver_flip(depth)
                lr = self.ver_flip(lr)
            if np.random.uniform(0.0, 1.0) < p_hor_flip:
                image = self.hor_flip(image)
                depth = self.hor_flip(depth)
                lr = self.hor_flip(lr)

        if not self.train:
            name = self.depths[idx].split("/")[-1].split('\\')[-1]
            sample = {'guidance': image, 'lr': lr, 'gt': depth, 'min': minn, 'max': maxx, 'name': name}
        else:
            sample = {'guidance': image, 'lr': lr, 'gt': depth, 'min': minn, 'max': maxx}

        return sample

    def __len__(self):
        return len(self.depths)


if __name__ == "__main__":

    dataset = Middlebury_Train(root_dir='/home/szdl/lsz/DMSG-train/train', scale=4, train=True, transform=trans.Compose([trans.ToTensor()]))
    for idx, data in enumerate(dataset):
        print(len(dataset))

        print('guidance', data['guidance'].shape, data['guidance'].max(), data['guidance'].min())
        print('lr', data['lr'].shape, data['lr'].max(), data['lr'].min())
        print('gt', data['gt'].shape, data['gt'].max(), data['gt'].min())
        break