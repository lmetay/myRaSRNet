
import torch.nn as nn

class DS_layer(nn.Module):
    def __init__(self, in_d, out_d, stride, output_padding, n_class):
        super(DS_layer, self).__init__()

        self.dsconv1 = nn.Sequential(nn.ConvTranspose2d(in_d, out_d, kernel_size=3, padding=1, stride=stride,
                                                       output_padding=output_padding),
                                    nn.BatchNorm2d(out_d),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout2d(p=0.2),
                                    nn.ConvTranspose2d(out_d, out_d, kernel_size=3, padding=1,stride=stride,
                                         output_padding=output_padding)
                                    )
        self.dsconv2 = nn.Sequential(nn.ConvTranspose2d(in_d, out_d, kernel_size=3, padding=1, stride=stride,
                                                       output_padding=output_padding),
                                    nn.BatchNorm2d(out_d),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout2d(p=0.2),
                                    nn.ConvTranspose2d(out_d, out_d, kernel_size=3, padding=1, stride=4,
                                         output_padding=3)
                                    )
        self.outconv = nn.ConvTranspose2d(out_d, n_class, kernel_size=3, padding=1)

    def forward(self, input):
        b,c,h,w = input.size()
        if h == 64:                        
            x = self.dsconv1(input)
        if h == 32:
            x = self.dsconv2(input)
        x = self.outconv(x)
        return x

