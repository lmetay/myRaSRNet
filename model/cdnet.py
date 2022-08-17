
import torch
import torch.nn as nn
from .Encoder_resnet18 import mynet3
from .Decoder import Decoder
from .Seg_Head import SegHead
from .FeatureFusion import FeatureFusion
from .Deeply_supervised_layer import DS_layer
import matplotlib.pyplot as plt  

class CDNet(nn.Module):
    def __init__(self, in_c, num_filters,  nr_object_class):
        super(CDNet,self).__init__()
        self.in_c = in_c 
        self.num_filters = num_filters
        self.nr_object_class = nr_object_class
        self.backbone = mynet3(in_c=self.in_c, f_c=64, output_stride=32)
        self.decoder = Decoder(self.in_c, self.num_filters) 
        self.FeatureFusion = FeatureFusion(deep_c=[512,256,128], shallow_c=[256,128,64])
        self.SegHead = SegHead(self.nr_object_class)

        self.ds_lyr1 = DS_layer(64, 32, 2, 1, self.nr_object_class) 
        self.ds_lyr2 = DS_layer(128, 32, 2, 1, self.nr_object_class)  
        self.ds_ly = nn.ModuleList([self.ds_lyr1, self.ds_lyr2])  

    
    def forward(self, input1, input2):
        A_res1, A_res2, A_res3, A_res4 = self.backbone(input1)
        B_res1, B_res2, B_res3, B_res4 = self.backbone(input2)

        A_df = self.decoder(A_res1, A_res2, A_res3, A_res4)
        B_df = self.decoder(B_res1, B_res2, B_res3, B_res4)

        group_feat = self.FeatureFusion(A_df,B_df)
        segDImap, segmap1, segmap2 = self.SegHead(group_feat[0],group_feat[1],group_feat[2])

        ds1 = self.ds_ly[0](torch.abs(A_res1 - B_res1))
        ds2 = self.ds_ly[1](torch.abs(A_res2 - B_res2))
       
        return segDImap, segmap1, segmap2, ds1, ds2

