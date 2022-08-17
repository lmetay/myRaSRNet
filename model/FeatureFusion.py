 
import torch
import torch.nn as nn
from .Decoder import Decoder
import torch.nn.functional as F
from osgeo import gdal
import os
import matplotlib.pyplot as plt
import numpy as np

torch.set_default_tensor_type(torch.DoubleTensor)

class FeatureFusion(nn.Module):
    def __init__(self,deep_c=[512,256,128], shallow_c=[256,128,64]):
        super(FeatureFusion,self).__init__()
        self.deep_c = deep_c
        self.shallow_c = shallow_c
        self.diff_stream_FF= diff_stream_FF(self.deep_c, self.shallow_c)
        self._init_weight()

    def forward(self, input1, input2):
        ff_A, ff_B, diff = self.diff_stream_FF(input1, input2)
        return ff_A, ff_B, diff
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class diff_stream_FF(nn.Module):
    def __init__(self,deep_c=[512,256,128], shallow_c=[256,128,64]): 
        super(diff_stream_FF,self).__init__()
        self.main_stream_FF = main_stream_FF()
        self.deep_c = deep_c
        self.shallow_c = shallow_c

        self.conv_out = nn.Sequential(
                            nn.Conv2d(in_channels=self.shallow_c[2], out_channels=self.shallow_c[2], 
                                      kernel_size=3, stride=1, padding=1, bias=False),
                            nn.BatchNorm2d(self.shallow_c[2]),
                            nn.ReLU(inplace=True),
        )

    def forward(self, image1_feat, image2_feat):
        I1_12, I1_23, I1_34 = self.main_stream_FF(image1_feat[0],image1_feat[1],
                                                    image1_feat[2],image1_feat[3])
        I2_12, I2_23, I2_34 = self.main_stream_FF(image2_feat[0],image2_feat[1],
                                                    image2_feat[2],image2_feat[3])

        Diff_feat = abs(I1_34 - I2_34)
        output = self.conv_out(Diff_feat)

        return I1_34, I2_34, output
    
    
class main_stream_FF(nn.Module):
    def __init__(self,deep_c=[512,256,128], shallow_c=[256,128,64]):
        super(main_stream_FF, self).__init__()
        self.deep_c = deep_c
        self.shallow_c = shallow_c
        self.f1 = FeatureFusionUnit(deep_c=self.deep_c[0], shallow_c=self.shallow_c[0], is_upsample=True)
        self.f2 = FeatureFusionUnit(deep_c=self.deep_c[1], shallow_c=self.shallow_c[1], is_upsample=True)
        self.f3 = FeatureFusionUnit(deep_c=self.deep_c[2], shallow_c=self.shallow_c[2], is_upsample=True)

    def forward(self, d1, d2, d3, d4):
        df1_2 = self.f1(d1, d2)
        df2_3 = self.f2(df1_2, d3)
        df3_4 = self.f3(df2_3, d4)
        return df1_2, df2_3, df3_4
        


class FeatureFusionUnit(nn.Module):
    def __init__(self, deep_c, shallow_c, is_upsample):
        super(FeatureFusionUnit, self).__init__()
        self.is_upsample = is_upsample
        self.Conv_deep = nn.Sequential(
                         nn.Conv2d(in_channels=deep_c, out_channels=shallow_c, kernel_size=3, padding=1, bias=False),
                         nn.BatchNorm2d(shallow_c),
                         nn.ReLU(inplace=True),
        )
        self.Conv_shallow = nn.Sequential(
                            nn.Conv2d(in_channels=shallow_c, out_channels=shallow_c, kernel_size=1, bias=False),
                            nn.BatchNorm2d(shallow_c),
                            nn.ReLU(inplace=True),
        )
        self.Conv_cat = nn.Sequential(
                            nn.Conv2d(in_channels=2*shallow_c, out_channels=shallow_c, kernel_size=1, bias=False),
                            nn.BatchNorm2d(shallow_c),
                            nn.ReLU(inplace=True),
        )

    def forward(self, deep_feat, shallow_feat):
        if self.is_upsample:
            deep_feat = F.interpolate(deep_feat, scale_factor=2, mode='bilinear', align_corners=True)

        deep_feat = self.Conv_deep(deep_feat)
        shallow_feat = self.Conv_shallow(shallow_feat)
        cat_feat = torch.cat((deep_feat, shallow_feat), dim=1)
        output = self.Conv_cat(cat_feat)

        return output


class Diff_FeatureFusionUnit(nn.Module):
    def __init__(self, deep_c, shallow_c, is_upsample):
        super(Diff_FeatureFusionUnit, self).__init__()
        self.is_upsample = is_upsample
        self.Conv_deep = nn.Sequential(
                         nn.Conv2d(in_channels=deep_c, out_channels=shallow_c, kernel_size=3, padding=1, bias=False),
                         nn.BatchNorm2d(shallow_c),
                         nn.ReLU(inplace=True),
        )
        self.Conv_shallow = nn.Sequential(
                            nn.Conv2d(in_channels=2*shallow_c, out_channels=shallow_c, kernel_size=3, padding=1, bias=False),
                            nn.BatchNorm2d(shallow_c),
                            nn.ReLU(inplace=True),
        )
        self.Conv_cat = nn.Sequential(
                            nn.Conv2d(in_channels=2*shallow_c, out_channels=shallow_c, kernel_size=1, bias=False),
                            nn.BatchNorm2d(shallow_c),
                            nn.ReLU(inplace=True),
        )

    def forward(self, deep_feat, shallow_feat):
        if self.is_upsample:
            deep_feat = F.interpolate(deep_feat, scale_factor=2, mode='bilinear', align_corners=False)

        deep_feat = self.Conv_deep(deep_feat)
        shallow_feat = self.Conv_shallow(shallow_feat)
        cat_feat = torch.cat((deep_feat, shallow_feat), dim=1)
        output = self.Conv_cat(cat_feat)

        return output



