import torch
import torch.nn.functional as F
import torch.nn as nn

import numpy as np
import pdb, os, argparse
from scipy import misc
#import imageio
import cv2

from model.CPD_models import JRBM
from model.CPD_ResNet_models import JRBM_ResNet
from data import test_dataset
#from featview import vis_feat,vis_alpha

import glob2

import scipy.io as sio # mat
#import matplotlib.pyplot as plt
eps = 2.2204e-16
import shutil

#foreground dilate,background erode
#result = 1-dilate+erode
def parameter():
    p = {}
    p['gtThreshold'] = 0.5
    p['beta'] = np.sqrt(0.3)
    p['thNum'] = 100
    p['thList'] = np.linspace(0, 1, p['thNum'])

    return p


def im2double(im):
    return cv2.normalize(im.astype('float'),
                         None,
                         0.0, 1.0,
                         cv2.NORM_MINMAX)


def prCount(gtMask, curSMap, p):
    gtH, gtW = gtMask.shape[0:2]
    algH, algW = curSMap.shape[0:2]

    if gtH != algH or gtW != algW:
        curSMap = cv2.resize(curSMap, (gtW, gtH))

    gtMask = (gtMask >= p['gtThreshold']).astype(np.float32)
    gtInd = np.where(gtMask > 0)
    gtCnt = np.sum(gtMask)


    hitCnt = np.zeros((p['thNum'], 1), np.float32)
    algCnt = np.zeros((p['thNum'], 1), np.float32)

    for k, curTh in enumerate(p['thList']):
        thSMap = (curSMap >= curTh).astype(np.float32)
        hitCnt[k] = np.sum(thSMap[gtInd])
        algCnt[k] = np.sum(thSMap)

    prec = hitCnt / (algCnt+eps)
    recall = hitCnt / (gtCnt + 1e-10)

    return prec, recall


def PR_Curve(resDir, gtDir):
    p = parameter()
    beta = p['beta']
    gtImgs = glob2.iglob(gtDir + '/*.png')  ########

    prec = []
    recall = []
    i = 0
    for gtName in gtImgs:
        dir, name = os.path.split(gtName)
        mapName = os.path.join(resDir,name[:-4]+'.png')
        i +=1
        #print(mapName)
        curMap = im2double(cv2.imread(mapName, cv2.IMREAD_GRAYSCALE))
        #print('map:',curMap.shape)
        curGT = im2double(cv2.imread(gtName, cv2.IMREAD_GRAYSCALE))
        #print('gt:', curGT.shape)
        if curMap.shape[0] != curGT.shape[0]:
            print('====================')
            print('mapName',mapName)
            curMap = cv2.resize(curMap, (curGT.shape[1], curGT.shape[0]))

        curPrec, curRecall = prCount(curGT, curMap, p)
        #print('prec:',len(curPrec))

        prec.append(curPrec)
        recall.append(curRecall)

    #print('prec:',prec)
    #print('i:',i)
    prec = np.hstack(prec[:])
    print('=')
    recall = np.hstack(recall[:])

    prec = np.mean(prec, 1)
    recall = np.mean(recall, 1)

    # compute the max F-Score
    score = (1+beta**2)*prec*recall / (beta**2*prec + recall)
    curTh = np.argmax(score)
    curScore = np.max(score)
    res = {}
    res['prec'] = prec
    res['recall'] = recall
    res['curScore'] = curScore
    res['curTh'] = curTh
    res['fscore']=score


    return res


def MAE_Value(resDir, gtDir):
    p = parameter()
    gtThreshold = p['gtThreshold']

    gtImgs = glob2.iglob(gtDir + '/*.png') 

    MAE = []


    for gtName in gtImgs:
        dir, name = os.path.split(gtName)
        mapName= os.path.join(resDir,name[:-4]+'.png')        ######
        #print(os.path.join(resDir,name[:-4]+'.png'))

        #print(mapName)
        if os.path.exists(mapName) is False:
            mapName = mapName.replace('.png', '.jpg')
            if os.path.exists(mapName) is False:
                mapName = mapName.replace('.jpg','.bmp')

        curMap = im2double(cv2.imread(mapName, cv2.IMREAD_GRAYSCALE))
        #print(curMap.shape)
        curGT = im2double(cv2.imread(gtName, cv2.IMREAD_GRAYSCALE))
        #print(curGT.shape)
        curGT = (curGT >= gtThreshold).astype(np.float32)

        if curMap.shape[0] != curGT.shape[0]:
            #print('mapName:::',mapName)
            curMap = cv2.resize(curMap, (curGT.shape[1], curGT.shape[0]))

        diff = np.abs(curMap - curGT)

        MAE.append(np.mean(diff))

    return np.mean(MAE)


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--is_ResNet', type=bool, default=False, help='VGG or ResNet backbone')
opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

dataset_path = '/test/total/'
test_datasets = ['all/']


j=1
#model = CPD_VGG()
while(j<=1):
    model = JRBM(32)
    #model = JRBM_ResNet(32)
    model = nn.DataParallel(model)

    model.load_state_dict(torch.load('./ors_vgg/model-'+str(j)))
    model.cuda()
    model.eval()
    for dataset in test_datasets:
            
            save_pre = './results_vgg_ors/test_'+str(j)+'/'
            
            print('j=',j,'is_ResNet:',save_path1)

            image_root = dataset_path + dataset + '/images/'
            gt_root = dataset_path + dataset + '/gt/'
            edge_root = dataset_path + dataset + '/gt/'
            gt_back_root = dataset_path + dataset + '/gt/'
            test_loader = test_dataset(image_root, gt_root, edge_root,gt_back_root, opt.testsize)

            for i in range(test_loader.size):
                image, gt, edge,gt_back, name = test_loader.load_data()
                gt = np.asarray(gt, np.float32)
                gt /= (gt.max() + 1e-8)
                image = image.cuda()
                
                if not os.path.exists(save_pre):
                    os.makedirs(save_pre)
                 
                out1,out2,out3 = model(image)

                res = F.upsample(out3, size=gt.shape, mode='bilinear', align_corners=True)
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
              
                cv2.imwrite(save_pre+name, res*255)

            gtDir = '/test/total/all/gt/'
            mae = MAE_Value(save_pre, gtDir)
            pr = PR_Curve(save_pre, gtDir)
            FMeasureF = pr['curScore']
            print('epoch:',j,'max F:', pr['curScore'],'MAE:', mae)
            j = j+1

