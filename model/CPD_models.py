# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.HolisticAttention import HA
from model.vgg import B2_VGG
from model.BGM_PPM import BGModel,GGM,CAM_Module#,HAM
#from model.SIM import SIM

def upsample_add(*xs):
    y = xs[-1]
    for x in xs[:-1]:
        y = y + F.interpolate(x, size=y.size()[2:], mode='bilinear', align_corners=False)
    return y

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(BasicConv2d, self).__init__()
        
        self.basicconv = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, groups=groups, bias=bias),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.basicconv(x)

class HAM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(HAM, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            nn.Conv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            nn.Conv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            nn.Conv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 9), padding=(0, 4)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(9, 1), padding=(4, 0)),
            nn.Conv2d(out_channel, out_channel, 3, padding=9, dilation=9)
        )
        self.conv_cat = nn.Conv2d(5*out_channel, out_channel, 3, padding=1)
        self.conv_res = nn.Conv2d(in_channel, out_channel, 1)
        self.channel_attention = CAM_Module(out_channel)
        self.conv1 = nn.Conv2d(out_channel,out_channel,3,1,1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)
                m.bias.data.fill_(0)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)

        x_cat = torch.cat((x0, x1, x2, x3, x4), 1)
        x_cat = self.conv_cat(x_cat)
        x_cat = self.channel_attention(x_cat)
        x_cat = self.conv1(x_cat)

        x = self.relu(x_cat + self.conv_res(x))

        return x

class aggregation_add(nn.Module):
    def __init__(self, channel):
        super(aggregation_add, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(channel, channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3*channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               + self.conv_upsample3(self.upsample(x2)) * x3

        x3_2 = torch.cat((x3_1, self.conv_upsample4(self.upsample(self.upsample(x1_1))), self.conv_upsample5(self.upsample(x2_1))), 1)
        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x



class JRBM(nn.Module):
    def __init__(self, channel=32):
        super(JRBM, self).__init__()
        self.vgg = B2_VGG()
        self.agg1 = aggregation_add(channel)
        self.agg2 = aggregation_add(channel)
        self.ham3_1 = HAM(256, channel)
        self.ham4_1 = HAM(512, channel)
        self.ham5_1 = HAM(512, channel)
        self.bgm5_1 = BGModel(channel, 4, 2)
        self.bgm4_1 = BGModel(channel, 4, 1)
        self.bgm3_1 = BGModel(channel, 2, 1)
        

        self.ham3_2 = HAM(256, channel)
        self.ham4_2 = HAM(512, channel)
        self.ham5_2 = HAM(512, channel)
        self.bgm5_2 = BGModel(channel, 4, 2)
        self.bgm4_2 = BGModel(channel, 4, 1)
        self.bgm3_2 = BGModel(channel, 2, 1)
      

        self.HA = HA()
        self.glob = GGM(512)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)

        self.glob_vgg2 = nn.Sequential(
            nn.Conv2d(512 + 128, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, channel, 3, 1, 1)
        )
        self.conv3 = nn.Conv2d(channel, 1, 1)



    def forward(self, x):
        x1 = self.vgg.conv1(x)
        x2 = self.vgg.conv2(x1)
        x3 = self.vgg.conv3(x2)

        x3_1 = x3
        x4_1 = self.vgg.conv4_1(x3_1)
        x5_1 = self.vgg.conv5_1(x4_1)
        x_glob = self.glob(x5_1)        #out_channel=512

        x_edge = self.glob_vgg2(torch.cat((self.upsample8(x_glob),x2),1))   #out_channel=channel

        x3_1 = self.ham3_1(x3_1)
        x4_1 = self.ham4_1(x4_1)
        x5_1 = self.ham5_1(x5_1)
        x5_1 = self.bgm5_1(x_edge, x5_1)
        x4_1 = self.bgm4_1(x_edge, x4_1)
        x3_1 = self.bgm3_1(x_edge, x3_1)
        attention = self.agg1(x5_1, x4_1, x3_1)


        x3_2 = self.HA(attention.sigmoid(), x3)
        x4_2 = self.vgg.conv4_2(x3_2)
        x5_2 = self.vgg.conv5_2(x4_2)

        x3_2 = self.ham3_2(x3_2)
        x4_2 = self.ham4_2(x4_2)
        x5_2 = self.ham5_2(x5_2)

        x_edge_pre = self.conv3(x_edge)
        x5_2 = self.bgm5_2(x_edge,x5_2)
        x4_2 = self.bgm4_2(x_edge,x4_2)
        x3_2 = self.bgm3_2(x_edge,x3_2)

        detection = self.agg2(x5_2, x4_2, x3_2)


        return self.upsample4(attention),self.upsample2(x_edge_pre), self.upsample4(detection)
