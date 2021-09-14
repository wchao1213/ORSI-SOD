#!/usr/bin/python3
#coding=utf-8

import sys
import datetime

import torch.distributed as dist
from utils import clip_gradient, adjust_lr
#from apex.parallel import convert_syncbn_model
#from apex.parallel import DistributedDataParallel

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
#from tensorboardX import SummaryWriter
from lib import dataset
from model.CPD_models  import JRBM
from model.CPD_ResNet_models  import JRBM_ResNet
#from model.loss import LovaszHinge,CEL
import logging as logger
from lib.data_prefetcher import DataPrefetcher
from lib.lr_finder import LRFinder
import numpy as np
#import matplotlib.pyplot as plt
import os
from torch.autograd import Variable
#from apex import amp
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

TAG = "tmp"
SAVE_PATH = "tmp"
logger.basicConfig(level=logger.INFO, format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', \
                           filename="train_%s.log"%(TAG), filemode="w")


""" set lr """
def get_triangle_lr(base_lr, max_lr, total_steps, cur, ratio=1., \
        annealing_decay=1e-2, momentums=[0.95, 0.85]):
    first = int(total_steps*ratio)
    last  = total_steps - first
    min_lr = base_lr * annealing_decay

    cycle = np.floor(1 + cur/total_steps)
    x = np.abs(cur*2.0/total_steps - 2.0*cycle + 1)
    if cur < first:
        lr = base_lr + (max_lr - base_lr) * np.maximum(0., 1.0 - x)
    else:
        lr = ((base_lr - min_lr)*cur + min_lr*first - base_lr*total_steps)/(first - total_steps)
    if isinstance(momentums, int):
        momentum = momentums
    else:
        if cur < first:
            momentum = momentums[0] + (momentums[1] - momentums[0]) * np.maximum(0., 1.-x)
        else:
            momentum = momentums[0]

    return lr, momentum

BASE_LR = 1e-3
MAX_LR = 0.1
FIND_LR = False #True


def train(Dataset, Network):
    cfg    = Dataset.Config(channel=32,datapath='/data/wchao_train', savepath=SAVE_PATH, mode='train', batch=15, lr=1e-4, momen=0.9, decay=5e-4, epoch=100)#0.05
   
    data   = Dataset.Data(cfg)
    loader = DataLoader(data, batch_size=cfg.batch, shuffle=True, num_workers=8)#new add
    prefetcher = DataPrefetcher(loader)
    ## network
    net    = Network(32)
    net = nn.DataParallel(net)
#    lovaszhinge = LovaszHinge()
#    cel = CEL()
    net.train(True)
    net.cuda()
    
    params = net.parameters()
    optimizer = torch.optim.Adam(params, 1e-4)
    
    ## parameter
#    base, head = [], []
#    for name, param in net.named_parameters():
#        if 'bkbone' in name:
#            base.append(param)
#        else:
#            head.append(param)
#    optimizer   = torch.optim.SGD([{'params':base}, {'params':head}], lr=cfg.lr, momentum=cfg.momen, weight_decay=cfg.decay, nesterov=True)
#    net, optimizer = amp.initialize(net, optimizer, opt_level='O0')
#    sw          = SummaryWriter(cfg.savepath)
    global_step = 0

    db_size = len(loader)
    if FIND_LR:
        lr_finder = LRFinder(net, optimizer, criterion=None)
        lr_finder.range_test(loader, end_lr=50, num_iter=100, step_mode="exp")
        #plt.ion()
        lr_finder.plot()
        import pdb; pdb.set_trace()

    #training
    for epoch in range(cfg.epoch):
        prefetcher = DataPrefetcher(loader)
        batch_idx = -1
        image, mask, bg,edge = prefetcher.next()
        while image is not None:
            
            optimizer.zero_grad()
            batch_idx += 1
            global_step += 1

            out1, out2, out3 = net(image)
            
            label1 = F.interpolate(mask, size=out1.shape[2:], mode='bilinear')
            label2 = F.interpolate(mask, size=out3.shape[2:], mode='bilinear')
           
            
            edge = F.interpolate(edge, size=out2.shape[2:], mode='bilinear')
            
            
            loss1  = F.binary_cross_entropy_with_logits(out1, label1)
            loss2  = F.binary_cross_entropy_with_logits(out2, edge)
            loss3  = F.binary_cross_entropy_with_logits(out3, label2)
            
            
            loss   =loss1 + 0.7*loss2 + loss3 

            loss.backward()
#            with amp.scale_loss(loss, optimizer) as scale_loss:
#                scale_loss.backward()
            
            clip_gradient(optimizer, 0.5)
            optimizer.step()    
            
#            optimizer.step()
#            sw.add_scalar('lr'   , optimizer.param_groups[0]['lr'], global_step=global_step)
#            sw.add_scalars('loss', {'loss':loss.item(), 'loss1':loss1.item(), 'loss2':loss2.item(), 'loss3':loss3.item()}, global_step=global_step)
            if batch_idx % 50 == 0:
                msg = '%s | step:%d/%d/%d | lr=%.6f | loss=%.6f | loss1=%.6f | loss2=%.6f | loss3=%.6f '%(datetime.datetime.now(),  global_step, epoch+1, cfg.epoch, optimizer.param_groups[0]['lr'], loss.item(), loss1.item(), loss2.item(), loss3.item())
                print(msg)
                logger.info(msg)
            image, mask, bg,edge = prefetcher.next()

        if (epoch+1)<= 100 and (epoch+1)==cfg.epoch:
            torch.save(net.state_dict(), cfg.savepath+'/model-'+str(epoch+1))


if __name__=='__main__':
    train(dataset, JRBM)

