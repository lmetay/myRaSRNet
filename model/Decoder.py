


import torch
import torch.nn as nn
from osgeo import gdal
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from .RA import GloSpaRe_Unit, GloChaRe_Unit


class Decoder(nn.Module):
    def __init__(self, in_c=3, num_filters = [256,128,64,32]): 
        super(Decoder, self).__init__()
        self.in_c = in_c
        self.num_filters = num_filters

        self.dr2_s = GloSpaRe_Unit(num_in=256, num_mid=64, normalize=False)
        self.dr3_s = GloSpaRe_Unit(num_in=128, num_mid=32, normalize=False)
        self.dr4_s = GloSpaRe_Unit(num_in=64, num_mid=16, normalize=False)

        self.dr2_c = GloChaRe_Unit(num_in=16*16, num_mid=8*8, normalize=False)
        self.dr3_c = GloChaRe_Unit(num_in=32*32, num_mid=16*16, normalize=False)
        self.dr4_c = GloChaRe_Unit(num_in=64*64, num_mid=32*32, normalize=False)


        self.conv2 = nn.Sequential(nn.Conv2d(256*2, 256*2, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(256*2),
                                   nn.Dropout(0.5),
                                   nn.Conv2d(256*2, 256, kernel_size=1, stride=1, padding=0, bias=False),
                                   nn.BatchNorm2d(256), 
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(128*2, 128*2, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(128*2),
                                   nn.Dropout(0.5),
                                   nn.Conv2d(128*2, 128, kernel_size=1, stride=1, padding=0, bias=False),
                                   nn.BatchNorm2d(128), 
                                   nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(64*2, 64*2, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(64*2), 
                                   nn.Dropout(0.5),
                                   nn.Conv2d(64*2, 64, kernel_size=1, stride=1, padding=0, bias=False),
                                   nn.BatchNorm2d(64), 
                                   nn.ReLU(inplace=True))
        self._init_weight()
 
    def forward(self,f1,f2,f3,f4):
        
        d1_s = f4
        d2_s = self.conv2(torch.cat(( self.dr2_s(f3), self.dr2_c(f3) ), dim = 1))
        d3_s = self.conv3(torch.cat(( self.dr3_s(f2), self.dr3_c(f2) ), dim = 1))
        d4_s = self.conv4(torch.cat(( self.dr4_s(f1), self.dr4_c(f1) ), dim = 1))


        return d1_s,d2_s,d3_s,d4_s


    def _init_weight(self):        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0) 
    
