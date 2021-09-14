import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from torch.nn import functional as F
from torch.autograd import Variable
#from model.coordatt import CoordAtt
class BGModel(nn.Module):
    def __init__(self, channel,e1_stride, e2_stride):
        super(BGModel, self).__init__()
        self.relu = nn.ReLU()

        self.conv0 = nn.Conv2d(channel, channel, 5, e1_stride, 2)
        self.gamma = nn.Conv2d(channel, channel, 5, e2_stride, 2)

        self.conv1 = nn.Conv2d(channel, channel, 5, 1, 2)
        self.conv2 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.conv3 = nn.Conv2d(channel*2, channel, 3, 1, 1)
        self.conv4 = nn.Conv2d(2*channel, channel, 3, 1, 1)

        self.edge_pre = nn.Conv2d(channel, 1, 1)
        self.fea_pre = nn.Conv2d(channel, 1, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)      #feature 22,44,88    edge:88
                m.bias.data.fill_(0)                  #stride1    stride2   padding

    def forward(self, y, x):  # x= feature, y=edge
        x = x * F.sigmoid(x)
        x = self.conv1(x)

        y = y * F.sigmoid(y)
        y = self.relu(self.conv0(y))
        y = self.relu(self.gamma(y))

        edge = self.relu(self.conv2( x * y))
        e_pre = self.edge_pre(edge)
        fea = self.relu(self.conv3(torch.cat((x,y),1)))
        f_pre = self.fea_pre(fea)
        x = self.conv4(torch.cat((edge, fea),1))
        return x



class GGM(nn.Module):  # get global feature
    def __init__(self, in_channels):
        super(GGM, self).__init__()
        self.branch0 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.ReLU()
        )
        self.branch1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((3, 3)),
            nn.Conv2d(in_channels, in_channels, 1, 1),
            nn.ReLU()
        )
        self.branch2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((5, 5)),
            nn.Conv2d(in_channels, in_channels, 1, 1),
            nn.ReLU()
        )
        self.branch3 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, in_channels, 1, 1),
            nn.ReLU()
        )
        self.con = nn.Conv2d(in_channels * 4, in_channels, 3, 1, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)
                m.bias.data.fill_(0)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x1 = F.upsample(x1, x0.size()[2:], mode='bilinear', align_corners=True)
        x2 = self.branch2(x)
        x2 = F.upsample(x2, x0.size()[2:], mode='bilinear', align_corners=True)
        x3 = self.branch3(x)
        x3 = F.upsample(x3, x0.size()[2:], mode='bilinear', align_corners=True)
        x = self.con(torch.cat((x0, x1, x2, x3), 1))

        return x

class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


