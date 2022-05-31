"""Module consist of brain segmentation model including ResEncBlocks and ResDecBlocks.

BrainSeg class includes stack of encoding and decoding blocks, structured and utilized 
to predict brain networks in different domains."""

from torch import nn 
from blocks import ResEncBlocks, ResDecBlocks, ResEncBlocksPP, ResDecBlocksPP

class BrainSeg(nn.Module):
    """Class for implementing brain segmentation model by assembling encoding/decoding blocks.

    Args:
        i_channel (int): Size of input channel
        h_channel (int): Size of hidden channels
    """    
    def __init__(self, i_channel, h_channel):
        super(BrainSeg, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv3d(i_channel, h_channel[0], kernel_size=3),            
            nn.Sigmoid()
        )
        self.layer2 = ResEncBlocks(h_channel[0], h_channel[1])
        self.layer3 = ResEncBlocks(h_channel[1], h_channel[2])
        self.layer4 = ResEncBlocks(h_channel[2], h_channel[3])
        self.layer5 = nn.Dropout3d(p=.5)
        self.layer6 = ResDecBlocks(h_channel[3], h_channel[2])
        self.layer7 = ResDecBlocks(h_channel[2], h_channel[1])
        self.layer8 = nn.Dropout3d(p=.5)
        self.layer9 = ResDecBlocks(h_channel[1], h_channel[0])
        self.layer10 = nn.Sequential(
            nn.ConvTranspose3d(h_channel[0], i_channel, kernel_size=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        residual1 = self.layer1(x)
        residual2 = self.layer2(residual1)
        residual3 = self.layer3(residual2)
        out = self.layer4(residual3)
        out = self.layer5(out)
        out = self.layer6(out)
        out = residual3 + out.clone()
        out = self.layer7(out)
        out = self.layer8(out)
        out = residual2 + out.clone()
        out = self.layer9(out)
        out = residual1 + out.clone()
        return self.layer10(out)


class BrainSegPP(nn.Module):
    """BPARC++ : Class for implementing brain segmentation model by assembling encoding/decoding blocks.

    Args:
        i_channel (int): Size of input channel
        h_channel (int): Size of hidden channels
    """    
    def __init__(self, i_channel, h_channel):
        super(BrainSegPP, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv3d(i_channel, h_channel[0], kernel_size=3),            
            nn.GELU()
        )
        self.layer2 = ResEncBlocksPP(h_channel[0], h_channel[1])
        self.layer3 = ResEncBlocksPP(h_channel[1], h_channel[2])
        self.layer4 = ResEncBlocksPP(h_channel[2], h_channel[3])
        self.layer5 = nn.Dropout3d(p=.5)
        self.layer6 = ResDecBlocksPP(h_channel[3], h_channel[2])
        self.layer7 = ResDecBlocksPP(h_channel[2], h_channel[1])
        self.layer8 = nn.Dropout3d(p=.5)
        self.layer9 = ResDecBlocksPP(h_channel[1], h_channel[0])
        self.layer10 = nn.Sequential(
            nn.ConvTranspose3d(h_channel[0], i_channel, kernel_size=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        residual1 = self.layer1(x)
        residual2 = self.layer2(residual1)
        residual3 = self.layer3(residual2)
        out = self.layer4(residual3)
        out = self.layer5(out)
        out = self.layer6(out)
        out = residual3 + out.clone()
        out = self.layer7(out)
        out = self.layer8(out)
        out = residual2 + out.clone()
        out = self.layer9(out)
        out = residual1 + out.clone()
        return self.layer10(out)

