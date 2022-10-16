from turtle import forward
import torch.nn as nn

from cvConfig import yolo as archi

class Conv_BatchNorm_Relu(nn.Module):
    def __init__(self, in_ch, out_ch, **kwargs):
        super(Conv_BatchNorm_Relu, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, bias = False, **kwargs)
        self.bn = nn.BatchNorm2d(out_ch)
        self.leaky = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.leaky(x)
        return x


class YoLo(nn.Module):
    def __init__(self, archi = archi, in_ch = 3, **kwargs):
        super(YoLo, self).__init__()
        self.archi = archi
        self.in_ch = in_ch
        self.darknet = self._create_conv(self.archi)
        self.fcs = self._create_fcs()

    def _create_conv(self, architecture):
        layers = []
        in_ch = self.in_ch

        for archi in architecture:
            if type(archi) == tuple:
                layers.append(Conv_BatchNorm_Relu(in_ch, archi[1], kernel_size = archi[0], stride = archi[2], padding = archi[3]))
                self.in_ch = archi[1]
            if type(archi) == str:
                layers += [nn.MaxPool2d(kernel_size = 2, stride = 2)]
            if type(archi) == list:
                for layers in range(archi[-1]):
                    layer_first = archi[0]
                    layer_second = archi[1]

                    layers += [Conv_BatchNorm_Relu(in_ch, layer_first[1], kernel_size = layer_first[0], strides = layer_first[2], padding = layer_first[3])]
                    layers += [Conv_BatchNorm_Relu(archi[1], layer_second[1], kernel_size = layer_second[0], strides = layer_second[2], padding = layer_second[3])]

                    self.in_ch = layer_second[1]
            return nn.Sequential(*layers)

    def _create_fcs(self, nums = 1000):
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * 7 * 7, 8),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.Linear(8, 1000)
        )

    def forward(self, x):
        x = self.darknet(x)
        x = self.fcs(x)
        return x
      
      
      
      
      
      
      
      
      
      
      
      
 import torch.nn as nn
import torch

import cvConfig
archi = cvConfig.yolo
loss = cvConfig.loss

class YoLo(nn.Module):
    def __init__(self):
        super(YoLo,self).__init__()
        self.in_ch = 3
        self.conv = self._make_layer()
        
    def _make_layer(self):
        layers = []
        for layer in archi:
            if isinstance(layer, str):
                layers.append(nn.MaxPool2d(kernel_size = 2, stride = 2, ))
            else:
                kernel_size = layer[0]
                channels = layer[1]
                stride = layer[2]
                padding = layer[3]
                layers.append(nn.Conv2d(self.in_ch, channels, kernel_size = kernel_size, stride = stride, padding = padding))
                self.in_ch = channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = nn.Linear(x.shape[0], 4096)
        x = nn.Linear(4096, loss["guesser"] * loss["qutity"] + loss["num_cls"])
        return x

real = YoLo()
rng = torch.rand(4,3,448,448)
result = real(rng)   
