import torch
import torch.nn as nn
import torchvision.models as models
from model.BGM_PPM import BGModel,GGM,CAM_Module#,HAM
from model.HolisticAttention import HA
from model.ResNet import B2_ResNet
import torch.nn.functional as F

def upsample_add(*xs):
    y = xs[-1]
    for x in xs[:-1]:
        y = y + F.interpolate(x, size=y.size()[2:], mode='bilinear', align_corners=False)
    return y
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


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


class JRBM_ResNet(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32):
        super(JRBM_ResNet, self).__init__()
        self.resnet = B2_ResNet()
        self.agg1 = aggregation_add(channel)
        self.agg2 = aggregation_add(channel)

        # self.tran3_1 = BasicConv2d(1024, 512, 1, 1)
        self.tran4_1 = BasicConv2d(2048, 512, 1, 1)
        # self.tran3_2 = BasicConv2d(1024, 512, 1, 1)
        # self.tran4_2 = BasicConv2d(2048, 512, 1, 1)

        self.ham2_1 = HAM(512, channel)
        self.ham3_1 = HAM(1024, channel)
        self.ham4_1 = HAM(2048, channel)
        self.bgm4_1 = BGModel(channel, 4, 2)
        self.bgm3_1 = BGModel(channel, 4, 1)
        self.bgm2_1 = BGModel(channel, 2, 1)

        self.ham2_2 = HAM(512, channel)
        self.ham3_2 = HAM(1024, channel)
        self.ham4_2 = HAM(2048, channel)
        self.bgm4_2 = BGModel(channel, 4, 2)
        self.bgm3_2 = BGModel(channel, 4, 1)
        self.bgm2_2 = BGModel(channel, 2, 1)
        

        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.glob = GGM(512)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)

        self.glob_vgg2 = nn.Sequential(
            nn.Conv2d(512 + 64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, channel, 3, 1, 1)
        )
        self.conv3 = nn.Conv2d(channel, 1, 1)

        self.HA = HA()
        if self.training:
            self.initialize_weights()

    def forward(self, x):

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)  
        x_get_b = x
        
        x = self.resnet.maxpool(x)  
        x1 = self.resnet.layer1(x)  
        x1_1 = x1
        x2 = self.resnet.layer2(x1)  

        x2_1 = x2
        x3_1 = self.resnet.layer3_1(x2_1)  
        x4_1 = self.resnet.layer4_1(x3_1)  

        x_glob = self.glob(self.tran4_1(x4_1))        

        x_edge = self.glob_vgg2(torch.cat((self.upsample2(self.upsample8(x_glob)),x_get_b),1))   


        x2_1 = self.ham2_1(x2_1)
        x3_1 = self.ham3_1(x3_1)
        x4_1 = self.ham4_1(x4_1)

        x4_1 = self.bgm4_1(x_edge, self.upsample2(x4_1))
        x3_1 = self.bgm3_1(x_edge, self.upsample2(x3_1))
        x2_1 = self.bgm2_1(x_edge, self.upsample2(x2_1))
        attention_map = self.agg1(x4_1, x3_1, x2_1)  #1*80*80
        attention_map = F.interpolate(attention_map, size=x2.shape[2:], mode='bilinear')#1*40*40

        x2_2 = self.HA(attention_map.sigmoid(), x2)#512*40*40
        x3_2 = self.resnet.layer3_2(x2_2)  # 1024 x 20 x 20
        x4_2 = self.resnet.layer4_2(x3_2)  # 2048 x 10 x 10
        # x3_2 = self.tran3_2(x3_2)
        # x4_2 = self.tran4_2(x4_2)
        x2_2 = self.ham2_2(x2_2)
        x3_2 = self.ham3_2(x3_2)
        x4_2 = self.ham4_2(x4_2)

        x_edge_pre = self.conv3(x_edge)
        x4_2 = self.bgm4_2(x_edge,self.upsample2(x4_2))
        x3_2 = self.bgm3_2(x_edge,self.upsample2(x3_2))
        x2_2 = self.bgm2_2(x_edge,self.upsample2(x2_2))
        detection_map = self.agg2(x4_2, x3_2, x2_2)

        return self.upsample4(attention_map),self.upsample2(x_edge_pre), self.upsample8(detection_map)#return self.upsample(attention_map), self.upsample(detection_map)

    def initialize_weights(self):
        res50 = models.resnet50(pretrained=True)
        pretrained_dict = res50.state_dict()
        all_params = {}
        for k, v in self.resnet.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet.state_dict().keys())
        self.resnet.load_state_dict(all_params)
