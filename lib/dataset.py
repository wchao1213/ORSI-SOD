#!/usr/bin/python3
#coding=utf-8

import os
import os.path as osp
import cv2
import torch
import numpy as np
try:
    from . import transform
except:
    import transform

from torch.utils.data import Dataset

class Config(object):
    def __init__(self, **kwargs):
        self.kwargs    = kwargs
        print('\nParameters...')
        for k, v in self.kwargs.items():
            print('%-10s: %s'%(k, v))

            #raise ValueError
        self.mean = np.array([[[0.485*256, 0.456*256, 0.406*256]]])
        self.std = np.array([[[0.229*256, 0.224*256, 0.225*256]]])

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        else:
            return None


class Data(Dataset):
    def __init__(self, cfg):
        self.samples = []
        lines = os.listdir(os.path.join(cfg.datapath, 'images'))
        for line in lines:
            imagepath = os.path.join(cfg.datapath, 'images', line)
            maskpath = os.path.join(cfg.datapath, 'gt', line[:-4]+'.png')
            
            bgpath = os.path.join(cfg.datapath, 'bg', line[:-4]+'.png')
            edgepath = os.path.join(cfg.datapath, 'edge', line[:-4]+'.png') #'ed'+
            #edgepath = os.path.join(cfg.datapath, 'edge', 'ed'+line[:-4]+'.png') #'ed'+

            self.samples.append([imagepath, maskpath, bgpath,edgepath])

        if cfg.mode == 'train':
            self.transform = transform.Compose( transform.Normalize(mean=cfg.mean, std=cfg.std),
                                                transform.Resize(352, 352),
                                                transform.RandomHorizontalFlip(),
                                                transform.RandomCrop(320,320),
                                                transform.ToTensor())
        elif cfg.mode == 'test':
            self.transform = transform.Compose( transform.Normalize(mean=cfg.mean, std=cfg.std),
                                                transform.Resize(320, 320),
                                                transform.ToTensor())
        else:
            raise ValueError

    def __getitem__(self, idx):
        imagepath, maskpath, bgpath,edgepath = self.samples[idx]
        image               = cv2.imread(imagepath).astype(np.float32)[:,:,::-1]
        mask                = cv2.imread(maskpath).astype(np.float32)[:,:,::-1]
        #print(edgepath)
        bg                  = cv2.imread(bgpath).astype(np.float32)[:,:,::-1]
        edge                = cv2.imread(edgepath).astype(np.float32)[:,:,::-1]
        H, W, C             = mask.shape
        image, mask, bg,edge     = self.transform(image, mask, bg,edge)
        return image, mask, bg,edge, (H, W), maskpath.split('/')[-1], bgpath.split('/')[-1], edgepath.split('/')[-1]

    def __len__(self):
        return len(self.samples)



if __name__=='__main__':
    import matplotlib.pyplot as plt
    plt.ion()

    cfg  = Config(mode='train', datapath='./data/DUTS')
    data = Data(cfg)
    for i in range(100):
        image, depth, mask = data[i]
        image = image.permute(1,2,0).numpy()*cfg.std + cfg.mean
        mask  = mask.numpy()
        plt.subplot(121)
        plt.imshow(np.uint8(image))
        plt.subplot(122)
        plt.imshow(mask)
        input()
