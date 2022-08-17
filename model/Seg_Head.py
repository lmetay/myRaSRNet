
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt  

def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)

def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
            conv3x3(in_planes, out_planes, stride),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            )

class SegHead(nn.Module):
    def __init__(self, nr_object_class=1):
        super(SegHead, self).__init__()
        self.nr_object_class = nr_object_class
        self.object_feat_head = nn.Conv2d(64, 1, kernel_size=1,stride=1,padding=0)  
        self.upconv = conv3x3_bn_relu(64, 32, 1)
        self.object_head = nn.Sequential(
                    conv3x3_bn_relu(32, 16, 1),
                    nn.Conv2d(16, self.nr_object_class-1, kernel_size=1, bias=True),
                    nn.Sigmoid()  
                )
        self._init_weight()
    
    def forward(self, input1, input2, input3):

        segmap1 = self.object_feat_head(input1)
        segmap2 = self.object_feat_head(input2)
        input3 = F.interpolate(input3, scale_factor=2, mode='bilinear', align_corners=False)
        input3 = self.upconv(input3)
        input3 = F.interpolate(input3, scale_factor=2, mode='bilinear', align_corners=False)
        segDImap = self.object_head(input3)
        return segDImap,segmap1, segmap2
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)# kaiming 初始化
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
