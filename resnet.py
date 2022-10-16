import torch
import torch.nn as nn 

import cvConfig as cvConfig

class ResNet(nn.Module):
    def __init__(self, config):
        super(ResNet,self).__init__()
        self.config = config
        self.head = nn.Sequential(nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3),
                                  nn.BatchNorm2d(64),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1),
                                 )
        self.blocks = self.block()
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        print(self.blocks)

    def block(self):  
        blocks = []
        for num in range(self.config['stage_num']):
            blocks.append(Hourglass(self.config['stage' + str(num + 1)]['channels'], \
                self.config['stage' + str(num + 1)]['stride'], leader = True))
            for _ in range(self.config['stage' + str(num + 1)]['repeat'] - 1):
                blocks.append(Hourglass(self.config['stage' + str(num + 1)]['channels']))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.head(x)
        x = self.blocks(x)
        x = self.avg(x)
        # x = x.reshape(x.shape[0], -1)
        x = torch.flatten(x, 1)
        x = nn.Linear(len(x[0]), cvConfig.resnet['num_cls'])
        return x
        

class Hourglass(nn.Module):
    def __init__(self, out_ch, stride = 1, leader = False):
        super(Hourglass, self).__init__()
        self.expan = 4
        self.leader = leader
        self.relu = nn.ReLU()

        self.dimension_kernel = 1
        self.desire_kernel = 3
        self.stride = stride
        self.padding = 1
        self.in_ch = out_ch * self.expan
        if self.leader:
            if out_ch != 128:
                self.in_ch  = out_ch * 2
            else:
                self.in_ch = 64
            self.stride = 2
            self.dimension = nn.Sequential(nn.Conv2d(self.in_ch, out_ch * self.expan, kernel_size = 1, stride = self.stride),
                                           nn.BatchNorm2d(out_ch * self.expan),
                                          )
        self.shrink_dim = nn.Conv2d(self.in_ch, out_ch, kernel_size = self.dimension_kernel, stride = 1)
        self.shrink_bn = nn.BatchNorm2d(out_ch)

        self.desire = nn.Conv2d(out_ch, out_ch, kernel_size = self.desire_kernel, stride = self.stride, padding = self.padding)
        self.desire_bn = nn.BatchNorm2d(out_ch)

        self.expan_dim = nn.Conv2d(out_ch, out_ch * self.expan, self.dimension_kernel)
        self.expan_bn = nn.BatchNorm2d(out_ch * self.expan)


    def forward(self, x):
        if not self.leader:
            residual = x
        else:
            residual = x
            residual = self.dimension(residual)
        x = self.shrink_dim(x)
        x = self.shrink_bn(x)
        x = self.relu(x)

        x = self.desire(x)
        x = self.desire_bn(x)
        x = self.relu(x)

        x = self.expan_dim(x)
        x = self.expan_bn(x)
        x = self.relu(residual + x)
        return x


real = ResNet(cvConfig.resnet)

rng = torch.rand(cvConfig.resnet['batch'], 3, 448, 448)
result = real(rng)
