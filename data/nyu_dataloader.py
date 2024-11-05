from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from data.common import get_patch
import torchvision.transforms as trans

class NYU_v2_datset(Dataset):
    """NYUDataset."""

    def __init__(self, root_dir, scale=4, train=True, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            scale (float): dataset scale
            train (bool): train or test
            transform (callable, optional): Optional transform to be applied on a sample.
            
        """
        self.root_dir = root_dir
        self.transform = transform
        self.scale = scale
        self.train = train
        
        if train:
            self.depths = np.load('%s/train_depth_split.npy'%root_dir)
            self.images = np.load('%s/train_images_split.npy'%root_dir)
        else:
            self.depths = np.load('%s/test_depth.npy'%root_dir)
            self.images = np.load('%s/test_images_v2.npy'%root_dir)
        self.hor_flip = trans.RandomHorizontalFlip(1)
        self.ver_flip = trans.RandomVerticalFlip(1)
    def __len__(self):
        return self.depths.shape[0]

    def __getitem__(self, idx):
        depth = self.depths[idx]
        image = self.images[idx]
        if self.train:
            image, depth = get_patch(img=image, gt=np.expand_dims(depth,2), patch_size=256)
        h, w = depth.shape[:2]
        s = self.scale
        lr = np.array(Image.fromarray(depth.squeeze()).resize((w//s,h//s), Image.BICUBIC))

        if self.transform:
            image = self.transform(image).float()
            depth = self.transform(depth).float()
            lr = self.transform(np.expand_dims(lr,2)).float()
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
        
        # if s == 4:
        #     sample = {'guidance': image, 'lr': lr, 'gt': depth}
        # else:
        #     min = depth.min()
        #     max = depth.max()
        #     depth = (depth-min)/(max-min)
        #     minn = lr.min()
        #     maxx = lr.max()
        #     lr = (lr-minn)/(maxx-minn)
        #     image_max = image.min()
        #     image_min = image.max()
        #     image = (image-image_min)/(image_max-image_min)
        #     maxmin = np.array([max, min])
        #     imgmaxmin = np.array([image_max, image_min])
        #     sample = {'guidance': image, 'lr': lr, 'gt': depth, 'maxmin': maxmin, 'imgmaxmin': imgmaxmin}
        
        sample = {'guidance': image, 'lr': lr, 'gt': depth}
        
        return sample
