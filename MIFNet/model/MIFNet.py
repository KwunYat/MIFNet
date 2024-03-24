import torch.nn.functional as F
from model.MIFNetBlocks import *
from model.FasterNet import *
from model.GCN import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from thop import profile
import os

class MIFNet(nn.Module):
    def __init__(self, weights='./model/pretrain/fasternet_s-epoch.299-val_acc1.81.2840.pth', cfg='./model/cfg/fasternet_s.yaml'):
        super(MIFNet, self).__init__()

        self.Backbone = fasternet_s(weights='./model/pretrain/fasternet_s-epoch.299-val_acc1.81.2840.pth', cfg='./model/cfg/fasternet_s.yaml')

        self.efem_1 = EFEM(128, 1024, 128)
        self.efem_2 = EFEM(128, 512, 64)
        self.efem_3 = EFEM(128, 256, 32)
        
        self.Down1 = Downsample(64, 128, 2)
        self.Down2 = Downsample(32, 128, 4)
        
        self.gim = EAGCN(128, 1, (24,24))
        
        self.psp = PSPModule(1024, 512, sizes=(1, 2, 3, 6))
        self.ca = CA_Block(512, reduction=16)
        
        self.sr_4 = SRM(512)
        self.sr_3 = SRM(64)
        self.sr_2 = SRM(32)
        self.sr_1 = SRM(16)

        self.fia_1 = FAM(128, 128, 128)
        self.fia_2 = FAM(64, 64, 128)
        self.fia_3 = FAM(32, 32, 128)
                
        self.ACB_3 = ACBlock(64, 64)
        self.ACB_2 = ACBlock(64, 32)
        self.ACB_1 = ACBlock(64, 16)

        self.ps = nn.PixelShuffle(2)
        
        self.linear4 = nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1)
        self.linear3 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        self.linear2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linear1 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):

        feat0, feat1, feat2, feat3 = self.Backbone(x)
        # 128, 24, 24
        efem_1 = self.efem_1(feat3, feat0, [24, 24])
        # 64, 48, 48
        efem_2 = self.efem_2(feat2, feat0, [48, 48])
        # 32, 96, 96
        efem_3 = self.efem_3(feat1, feat0, [96, 96])
        # 128, 24, 24  
        efem_f2 = self.Down1(efem_2)
        efem_f3 = self.Down2(efem_3)
        
        # 128, 24, 24
        edge = efem_1 * efem_f2 * efem_f3
        # 512, 12, 12
        psp = self.psp(feat3)
        ca = self.ca(psp)
        # 512, 12, 12
        sr_4 = self.sr_4(ca, 512)
        # 128, 24, 24
        sr_4 = self.ps(sr_4)
        # 128, 24, 24
        gim = self.gim(sr_4, edge)
        # 64, 24, 24
        fia_1 = self.fia_1(efem_1, sr_4, gim)
        fia_1 = self.ACB_3(fia_1)
        # 64, 24, 24
        sr_3 = self.sr_3(fia_1, 64)
        # 64, 48, 48
        fia_2 = self.fia_2(efem_2, sr_3, gim)
        # 32, 48, 48
        fia_2 = self.ACB_2(fia_2)
        # 32, 48, 48
        sr_2 = self.sr_2(fia_2, 32)
        # 64, 96, 96
        fia_3 = self.fia_3(efem_3, sr_2, gim)
        # 16, 96, 96
        fia_3 = self.ACB_1(fia_3)
        # 16, 96, 96
        sr_1 = self.sr_1(fia_3, 16)
        # 4, 384, 384
        ps = self.ps(sr_1)

        if self.training == True:
            out4 = F.interpolate(self.linear4(ps), size=x.size()[2:], mode='bilinear')
            out3 = F.interpolate(self.linear3(sr_2), size=x.size()[2:], mode='bilinear')
            out2 = F.interpolate(self.linear2(sr_3), size=x.size()[2:], mode='bilinear')
            out1 = F.interpolate(self.linear1(sr_4), size=x.size()[2:], mode='bilinear')
            edge1 = F.interpolate(self.linear1(efem_1), size=x.size()[2:], mode='bilinear')
            edge2 = F.interpolate(self.linear2(efem_2), size=x.size()[2:], mode='bilinear')
            edge3 = F.interpolate(self.linear3(efem_3), size=x.size()[2:], mode='bilinear')
        else:
            out4 = F.interpolate(self.linear4(ps), size=x.size()[2:], mode='bilinear')

        if self.training == True:
            return torch.sigmoid(out4), torch.sigmoid(out3), torch.sigmoid(out2), torch.sigmoid(out1), torch.sigmoid(edge1), torch.sigmoid(edge2), torch.sigmoid(edge3)

        else:
            return torch.sigmoid(out4)


if __name__ == '__main__':
    import time
    os.environ['CUDA_VISIBLE_DEVICES'] = "5"
    net = MIFNet(weights = './model/pretrain/fasternet_s-epoch.299-val_acc1.81.2840.pth', cfg='./model/cfg/fasternet_s.yaml')
    inputs = torch.randn(1, 1, 384, 384)
    net = net.to('cuda')
    inputs = inputs.to('cuda')
    torch.cuda.synchronize()
    start_time = time.time()
    out1,out2,out3,out4,e1,e2,e3 = net(inputs)
    end_time = time.time()
    inference_time = end_time - start_time
    print('Latency = '+str(inference_time) + 's')
    flops, params = profile(net, (inputs,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')


