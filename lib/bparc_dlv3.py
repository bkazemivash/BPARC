"""Base DeepLab model with modifications to improve the performace for dense prediction. """

import torch
from torch import nn


class AtrousSpatialPyramidPooling(nn.Module):
    """Atrous Spatial Pyramid Pooling (ASPP) block
       Paper: https://arxiv.org/abs/1706.05587.

    Args:
        in_ch (int): channel size of input layer
        out_ch (int): channel size of hidden layers
    """
    def __init__(self, in_ch, out_ch) -> None:
        super(AtrousSpatialPyramidPooling, self).__init__()
        self.global_pooling = nn.Sequential(
            nn.AvgPool3d(kernel_size=3),
            nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU()                                                                       
        )
        self.stage1 = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch)           
        )    
        self.stage2 = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm3d(out_ch)             
        )        
        self.stage3 = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=3, dilation=3, bias=False),
            nn.BatchNorm3d(out_ch)   
        ) 
        
    def forward(self, x):
        _, _, H, W, D = x.shape
        h1 = self.global_pooling(x)
        h1 = nn.functional.interpolate(h1, size=(H, W, D), mode="trilinear", align_corners=False)
        h2 = self.stage1(x)
        h3 = self.stage2(x)
        h4 = self.stage3(x)
        return torch.cat((h1, h2, h3, h4), 1)


class FullyPreactivatedResidualUnit(nn.Module):
    """Class of residual unit with full pre-activation based on 
    https://arxiv.org/pdf/1603.05027.pdf and https://arxiv.org/abs/1512.03385.

    Args:
        in_ch (int): channel size of input layer
        out_ch (int): channel size of output layer
        mid_ch (int, optional): channel size of hidden layer. Defaults to None.
        downsample (int, optional): downsampling rate to control stride. Defaults to 1.
    """
    def __init__(self, in_ch, out_ch, downsample=1) -> None:
        super(FullyPreactivatedResidualUnit, self).__init__()
        self.stage1 = nn.Sequential(
            nn.BatchNorm3d(in_ch),
            nn.ReLU(),        
            nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=downsample, bias=False)
        )
        self.stage2 = nn.Sequential(
            nn.BatchNorm3d(out_ch),
            nn.ReLU(),        
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        )
        self.stage3 = nn.Sequential(
            nn.BatchNorm3d(out_ch),
            nn.ReLU(),        
            nn.Conv3d(out_ch, out_ch, kernel_size=1, bias=False)
        )        
        self.shortcut = nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=downsample, bias=False) if downsample > 1 else nn.Identity()

    def forward(self, x):
        out = self.stage1(x)
        out = self.stage2(out)
        out = self.stage3(out)
        identity = self.shortcut(x)
        out += identity
        return out


class StemUnit(nn.Module):
    def __init__(self, in_dim, hidden_dim, downsample=1):
        """Implementation of entry unit based on https://arxiv.org/abs/1812.01187

        Args:
            in_ch (int): channel size of input layer
            out_ch (int): channel size of output layer
            downsample (int, optional): downsampling rate to control stride. Defaults to 1.
        """
        super(StemUnit, self).__init__()
        self.stage1 = nn.Sequential(
            nn.BatchNorm3d(in_dim),
            nn.ReLU(),
            nn.Conv3d(in_dim, hidden_dim, kernel_size=3, padding=1, stride=downsample, bias=False),
            nn.MaxPool3d(2)
        )        

    def forward(self, x):
        return self.stage1(x)
