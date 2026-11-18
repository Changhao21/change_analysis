import torch
import numpy as np
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F
from utils.misc import initialize_weights

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class FCN(nn.Module):
    def __init__(self, in_channels=3, pretrained=True):
        super(FCN, self).__init__()
        resnet = models.resnet34(pretrained)
        newconv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        newconv1.weight.data[:, 0:3, :, :].copy_(resnet.conv1.weight.data[:, 0:3, :, :])
        if in_channels>3:
          newconv1.weight.data[:, 3:in_channels, :, :].copy_(resnet.conv1.weight.data[:, 0:in_channels-3, :, :])
          
        self.layer0 = nn.Sequential(newconv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        for n, m in self.layer3.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        self.head = nn.Sequential(nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False),
                                  nn.BatchNorm2d(128), nn.ReLU())
        initialize_weights(self.head)
                                  
    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes) )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)


class ResBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ChannelAttention(nn.Module):
    def __init__(self, embed_dim, factor, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Sequential(nn.Conv2d(embed_dim*factor, embed_dim*factor // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(embed_dim*factor // ratio, embed_dim*factor, 1, bias=False))
        self.fc2 = nn.Sequential(nn.Conv2d(embed_dim*factor, embed_dim*factor // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(embed_dim*factor // ratio, embed_dim*factor, 1, bias=False))

        self.fusion = nn.Sequential(
            nn.Conv2d(embed_dim*factor,embed_dim,3,1,1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU()
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc1(self.avg_pool(x))
        max_out = self.fc2(self.max_pool(x))
        att1 = self.sigmoid(avg_out)
        att2 = self.sigmoid(max_out)

        x1 = x * att1
        x2 = x * att2

        x = self.fusion(x1+x2)
        return x


class SpatialAttention(nn.Module):
    def __init__(self,embed_dim):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, 5, padding=5 // 2, bias=False)
        self.conv2 = nn.Conv2d(2, 1, 7, padding=7 // 2, bias=False)
        self.conv3 = nn.Conv2d(2, 1, 9, padding=9 // 2, bias=False)

        self.fusion = nn.Sequential(
            nn.Conv2d(embed_dim,embed_dim,3,1,1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU()
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        f = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        att1 = self.sigmoid(self.conv1(x))
        att2 = self.sigmoid(self.conv2(x))
        att3 = self.sigmoid(self.conv3(x))

        x1 = f * att1
        x2 = f * att2
        x3 = f * att3

        f = self.fusion(x1+x2+x3)

        return f


class CAFM(nn.Module):
    def __init__(self, embed_dim,factor):
        super().__init__()
        self.channel_attention = ChannelAttention(embed_dim,factor)
        self.spatial_attention = SpatialAttention(embed_dim)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)

        return x


class CSAM(nn.Module):
    '''Spatial reasoning module'''

    # codes from DANet 'Dual attention network for scene segmentation'
    def __init__(self, in_dim):
        super().__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.project = nn.Conv2d(in_dim,in_dim,1)

        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

        self.fusion = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 3, 1, 1),
            nn.BatchNorm2d(in_dim),
            nn.ReLU()
        )

    def forward(self, x):
        ''' inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW) '''
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.project(out)

        out = x + self.gamma * out

        out = self.fusion(out)

        return out


class SSCDl(nn.Module):
    def __init__(self, in_channels=3, num_classes=7):
        super(SSCDl, self).__init__()
        self.FCN = FCN(in_channels, pretrained=True)

        self.couple_seg1 = CAFM(128, 2)
        self.couple_seg2 = CAFM(128, 2)
        self.couple_CD = CAFM(128, 2)

        self.sa1 = CSAM(128)
        self.sa2 = CSAM(128)
        self.sa3 = CSAM(128)

        self.classifier1 = nn.Conv2d(128, num_classes, kernel_size=1)
        self.classifier2 = nn.Conv2d(128, num_classes, kernel_size=1)

        self.classifierCD = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Conv2d(64, 1, kernel_size=1))
            
        initialize_weights(self.couple_seg1,self.couple_seg2,self.couple_CD,self.sa1,self.sa2,self.sa3, self.classifier1,self.classifier2, self.classifierCD)
    
    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes) )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def base_forward(self, x):

        x = self.FCN.layer0(x) #size:1/4

        x = self.FCN.maxpool(x) #size:1/4

        x = self.FCN.layer1(x) #size:1/4

        x = self.FCN.layer2(x) #size:1/8

        x = self.FCN.layer3(x) #size:1/16

        x = self.FCN.layer4(x)

        x = self.FCN.head(x) #8倍下采样

        return x
    
    def forward(self, x1, x2):
        x_size = x1.size()
        
        x1 = self.base_forward(x1)
        x2 = self.base_forward(x2)


        x = torch.cat([x1,x2], 1)

        x1 = self.couple_seg1(x)
        x2 = self.couple_seg2(x)
        diff = self.couple_CD(x)

        x1 = self.sa1(x1)
        x2 = self.sa2(x2)
        diff = self.sa3(diff)

        change = self.classifierCD(diff)

        out1 = self.classifier1(x1)
        out2 = self.classifier2(x2)
        
        return F.interpolate(change, x_size[2:], mode='bilinear'), F.interpolate(out1, x_size[2:], mode='bilinear'), F.interpolate(out2, x_size[2:], mode='bilinear')


# x = torch.randn((5,3,256,256))
#
# net = SSCDl(in_channels=3,num_classes=7)
# cp,m1,m2 = net(x,x)
# print(cp.shape,m1.shape,m2.shape)


