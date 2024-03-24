import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import kornia

def weight_init(module):
    for n, m in module.named_children():
        print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        else:
            m.initialize()

class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(Downsample, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)
        self.conv = ConvBlock(in_channel, out_channel, kernel_size=1, stride=1, padding=0)        
    def forward(self, x):
        x = self.maxpool(x)
        x = self.conv(x)
        return x


class ACBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super (ACBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), stride=1, padding=[0, 1])
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), stride=1, padding=[1, 0])
        self.relu = nn.ReLU()

    def forward(self, input_image):
        layer1 = self.conv1(input_image)
        layer2 = self.conv2(input_image)
        layer3 = self.conv3(input_image)
        layer = layer1 + layer2 + layer3
        out = self.relu(layer)

        return out

class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()
 
    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)
 
    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)

class CA_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CA_Block, self).__init__()
        
        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel//reduction, kernel_size=1, stride=1, bias=False)
 
        self.relu   = nn.ReLU()
        self.bn     = nn.BatchNorm2d(channel//reduction)
 
        self.F_h = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)
        self.F_w = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)
 
        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()
 
    def forward(self, x):
        # b, c, h, w
        _, _, h, w = x.size()
        
        # b, c, h, w => # b, c, h, 1 => b, c, 1, h
        x_h = torch.mean(x, dim = 3, keepdim = True).permute(0, 1, 3, 2)
        
        # b, c, h, w => # b, c, 1, w
        x_w = torch.mean(x, dim = 2, keepdim = True)
        
        # b, c, 1, w cat b, c, 1, h => b, c, 1, h+w
        # b, c, 1, h+w => b, c/r, 1, h+w
        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))
 
        # b, c/r, 1, h+w => b, c/r, 1, h & b, c/r, 1, w
        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)
 
        # b, c/r, 1, h => b, c/r, h, 1 => b, c, h, 1
        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        
        # b, c/r, 1, w => b, c, 1, w
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))
           
        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        return out


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)





class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        return self.relu(self.bn(x))

class EFEM(nn.Module):
    def __init__(self, in_channel_1, in_channel_2, out_channel):
        super(EFEM, self).__init__()
        self.reduce1 = ConvBlock(in_channel_1, out_channel, kernel_size=1, stride=1, padding=0)
        self.reduce2 = ConvBlock(in_channel_2, out_channel, kernel_size=1, stride=1, padding=0)
        self.block = nn.Sequential(
            ConvBlock(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            ConvBlock(out_channel, out_channel,  kernel_size=3, stride=1, padding=1))

    def forward(self, x2, x1, scale):
        x1 = self.reduce1(x1)
        x2 = self.reduce2(x2)
        x3 = F.interpolate(x2, size=x1.size()[-2:], mode='bilinear')
        blur = x1 - kornia.filters.BoxBlur((5, 5))(x3)
        out = self.block(blur)
        out = F.interpolate(out, size=scale, mode='bilinear')
        return out

class SRM(nn.Module):
    def __init__(self, in_channel):
        super(SRM, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.conv2 = nn.Conv2d(in_channel, in_channel*2, kernel_size=3, stride=1, padding=1)

    def forward(self, x, in_channel):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True) #128
        out2 = self.conv2(out1)
        w, b = out2[:, :in_channel, :, :], out2[:, in_channel:, :, :]

        return F.relu(w * out1 + b, inplace=True)

    def initialize(self):
        weight_init(self)

class FAM(nn.Module):
    def __init__(self, in_channel_left, in_channel_down, in_channel_right):
        super(FAM, self).__init__()
        self.conv0 = nn.Conv2d(in_channel_left, 64, kernel_size=3, stride=1, padding=1)
        self.bn0   = nn.BatchNorm2d(64)
        self.conv1 = nn.Conv2d(in_channel_down, 64, kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channel_right, 64, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(64)

        self.conv_d1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv_d2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv_l = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64*3, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)


    def forward(self, left, down, right):
        left = F.relu(self.bn0(self.conv0(left)), inplace=True) #256 channels
        down = F.relu(self.bn1(self.conv1(down)), inplace=True) #256 channels
        right = F.relu(self.bn2(self.conv2(right)), inplace=True) #256

        down_1 = self.conv_d1(down)

        if down_1.size()[2:] != left.size()[2:]:
            down_1 = F.interpolate(down_1, size=left.size()[2:], mode='bilinear')

        down_2 = self.conv_d2(right)
        if down_2.size()[2:] != left.size()[2:]:
            down_2 = F.interpolate(down_2, size=left.size()[2:], mode='bilinear')


        out = torch.cat((left, down_1, down_2), dim=1)
        return F.relu(self.bn3(self.conv3(out)), inplace=True)


    def initialize(self):
        weight_init(self)

class GCF(nn.Module):
    def __init__(self, in_channel_left, in_channel_down):
        super(GCF, self).__init__()
        self.conv0 = nn.Conv2d(in_channel_left, 128, kernel_size=1, stride=1, padding=0)
        self.bn0   = nn.BatchNorm2d(128)
        self.conv1 = nn.Conv2d(in_channel_down, 128, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0)

    def forward(self, left, down):
        left = F.relu(self.bn0(self.conv0(left)), inplace=True) #256
        down = down.mean(dim=(2,3), keepdim=True)
        down = F.relu(self.conv1(down), inplace=True)
        down = torch.sigmoid(self.conv2(down))
        return left * down

    def initialize(self):
        weight_init(self)


