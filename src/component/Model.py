import torch
from torch import nn
from torch.nn import Conv2d, BatchNorm2d, LeakyReLU, Identity

from einops.layers.torch import Rearrange

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, use_bn=True):
        super().__init__()
        self.net=nn.Sequential(
            Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=not use_bn),
            BatchNorm2d(out_channels) if use_bn else Identity(),
            LeakyReLU(negative_slope=0.1) if use_bn else Identity()
        )

    def forward(self, x):
        return self.net(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, use_residual=True, num_repeats=1):
        super().__init__()
        layers=[]
        for _ in range(num_repeats):
            block=nn.Sequential(
                Conv2d(in_channels, in_channels//2, kernel_size=1),
                BatchNorm2d(in_channels//2),
                LeakyReLU(negative_slope=0.1),
                Conv2d(in_channels//2, in_channels, kernel_size=1),
                BatchNorm2d(in_channels),
                LeakyReLU(negative_slope=0.1)
            )
            layers.append(block)

        self.net=nn.ModuleList(layers)
        self.use_residual=use_residual
        self.num_repeats=num_repeats

    def forward(self, x):
        for layer in self.net:
            x = x + layer(x) if self.use_residual else layer(x)

        return x

class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes, num_anchors_per_scale):
        super().__init__()
        self.net=nn.Sequential(
            Conv2d(in_channels, 2*in_channels, kernel_size=3, padding=1),
            BatchNorm2d(2*in_channels),
            LeakyReLU(negative_slope=0.1),
            Conv2d(2*in_channels, (num_classes+5)*num_anchors_per_scale, kernel_size=1),
            Rearrange('b (a c) w h -> b a w h c', c=num_classes+5)
        )

    def forward(self, x):
        return self.net(x)

class Darknet(nn.Module):
    def __init__(self, in_channels, num_classes, num_anchors_per_scale=3):
        super().__init__()
        self.net=nn.ModuleList([
            CNNBlock(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            CNNBlock(in_channels=32, out_channels=64,kernel_size=3, stride=2, padding=1),
            ResidualBlock(in_channels=64, num_repeats=1),

            CNNBlock(in_channels=64, out_channels=128,kernel_size=3, stride=2, padding=1),
            ResidualBlock(in_channels=128, num_repeats=1),

            CNNBlock(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            ResidualBlock(in_channels=256, num_repeats=2),

            CNNBlock(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
            ResidualBlock(in_channels=512, num_repeats=2),

            CNNBlock(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=1),
            ResidualBlock(in_channels=1024, num_repeats=1),

            CNNBlock(in_channels=1024, out_channels=512, kernel_size=1),
            CNNBlock(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            ResidualBlock(in_channels=1024, use_residual=False, num_repeats=1),

            CNNBlock(in_channels=1024, out_channels=512, kernel_size=1),
            ScalePrediction(in_channels=512, num_classes=num_classes, num_anchors_per_scale=num_anchors_per_scale),

            CNNBlock(in_channels=512, out_channels=256, kernel_size=1),
            nn.Upsample(scale_factor=2),

            CNNBlock(in_channels=256+512, out_channels=256, kernel_size=1),
            CNNBlock(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            ResidualBlock(in_channels=512, use_residual=False, num_repeats=1),

            CNNBlock(in_channels=512, out_channels=256, kernel_size=1),
            ScalePrediction(in_channels=256, num_classes=num_classes, num_anchors_per_scale=num_anchors_per_scale),

            CNNBlock(in_channels=256, out_channels=128, kernel_size=1),
            nn.Upsample(scale_factor=2),

            CNNBlock(in_channels=128+256, out_channels=128, kernel_size=1),
            CNNBlock(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            ResidualBlock(in_channels=256, use_residual=False, num_repeats=1),

            CNNBlock(in_channels=256, out_channels=128, kernel_size=1),
            ScalePrediction(in_channels=128, num_classes=num_classes, num_anchors_per_scale=num_anchors_per_scale),
        ])

    def forward(self, x):
        outputs=[]
        route_connections=[]
        for idx, layer in enumerate(self.net):
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue
            x=layer(x)
            if isinstance(layer, ResidualBlock) and layer.num_repeats==2:
                route_connections.append(x)
            elif isinstance(layer, nn.Upsample):
                x=torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()

        return outputs

class YoloMimic(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.net_1 = Conv2d(in_channels, (num_classes+5)*3, kernel_size=3, stride=32, padding=1)
        self.net_2 = Conv2d(in_channels, (num_classes+5)*3, kernel_size=3, stride=16, padding=1)
        self.net_3 = Conv2d(in_channels, (num_classes+5)*3, kernel_size=3, stride=8, padding=1)

    def forward(self, x):
        x1 = self.net_1(x).reshape([-1, 3, 7, 7, self.num_classes+5])
        x2 = self.net_2(x).reshape([-1, 3, 14, 14, self.num_classes+5])
        x3 = self.net_3(x).reshape([-1, 3, 28, 28, self.num_classes+5])

        return x1, x2, x3
