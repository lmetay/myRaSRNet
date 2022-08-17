import torch
import torch.nn as nn




class GCN(nn.Module):

    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def forward(self, x):
        h = self.conv1(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1) 
        h = h + x
        h = self.conv2(self.relu(h)) #
        return h


class GloSpaRe_Unit(nn.Module):
    def __init__(self, num_in, num_mid, normalize=False):
        super(GloSpaRe_Unit, self).__init__()
        
        self.normalize = normalize
        self.num_s = int(2 * num_mid)
        self.num_n = int(1 * num_mid)

        self.conv_state = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.conv_proj = nn.Conv2d(num_in, self.num_n, kernel_size=1)
        self.gcn = GCN(num_state=self.num_s, num_node=self.num_n)
        self.conv_extend = nn.Conv2d(self.num_s, num_in, kernel_size=1, bias=False)
        self.blocker = nn.BatchNorm2d(num_in, eps=1e-04)


    def forward(self, x):
        B = x.size(0) # batchsize     
        x_state_reshaped = self.conv_state(x).view(B, self.num_s, -1) 
        x_proj_reshaped = self.conv_proj(x).view(B, self.num_n, -1)  
        x_rproj_reshaped = x_proj_reshaped 
        x_n_state = torch.matmul(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))  
        
        if self.normalize:
            x_n_state = x_n_state * (1. / x_state_reshaped.size(2))

        x_n_rel = self.gcn(x_n_state)                                  
        x_state_reshaped = torch.matmul(x_n_rel, x_rproj_reshaped)    
        x_state = x_state_reshaped.view(B, self.num_s, *x.size()[2:])  
        out = x + self.blocker(self.conv_extend(x_state))              
        return out


class GloChaRe_Unit(nn.Module):
    def __init__(self, num_in, num_mid, normalize=False):
        super(GloChaRe_Unit, self).__init__()

        self.normalize = normalize
        self.num_s = int(2 * num_mid)
        self.num_n = int(1 * num_mid)

        self.conv_state = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.conv_proj = nn.Conv2d(num_in, self.num_n, kernel_size=1)
        self.gcn = GCN(num_state=self.num_s, num_node=self.num_n)
        self.conv_extend = nn.Conv2d(self.num_s, num_in, kernel_size=1, bias=False)
        self.blocker = nn.BatchNorm2d(num_in, eps=1e-04)

    def forward(self, x):
        B, C, H0, W0 = x.size() 
        HW0 = H0 * W0
        x = x.view(B, C, HW0).permute(0, 2, 1)  
        x = x.contiguous().view(B, HW0, int(C/16), 16) 

        x_state_reshaped = self.conv_state(x).view(B, self.num_s, -1) 
        x_proj_reshaped = self.conv_proj(x).view(B, self.num_n, -1) 
        x_rproj_reshaped = x_proj_reshaped 
        x_n_state = torch.matmul(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))  

        if self.normalize:
            x_n_state = x_n_state * (1. / x_state_reshaped.size(2)) 

        x_n_rel = self.gcn(x_n_state)          
        x_state_reshaped = torch.matmul(x_n_rel, x_rproj_reshaped)    
        x_state = x_state_reshaped.view(B, self.num_s, *x.size()[2:])  
        x_extend =  self.conv_extend(x_state)                          
        out = x + self.blocker(x_extend)                              
        out = out.view(B, HW0, -1).permute(0, 2, 1)                    

        return out.view(B, C, H0, W0)                                  




