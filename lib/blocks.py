"""Module contains all model backbone blocks like  ResEncBlocks and ResDecBlocks.

ResDecBlocks class encodes input data into lower representation which is used by ResDecBlocks
which decodes the extracted features to reconstruct the output with same size as input."""

from torch import nn


class ResEncBlocks(nn.Module):
    """Class for implementing main block of encoding procedure to map input data to lower represeantation.

        Args:
            n_channels (int): channel size of input data
            o_channels (int): channel size of output data
            mid_channels (object:int, optional): channel size of hidden layers. Defaults to None.    

    """

    def __init__(self, n_channels, o_channels, mid_channels=None):
        super(ResEncBlocks, self).__init__()
        if not mid_channels:
            mid_channels = o_channels
        self.block1 = nn.Sequential(
            nn.Conv3d(n_channels, mid_channels, kernel_size=3),
            nn.BatchNorm3d(mid_channels),
            nn.Sigmoid()
        )
        self.block2 = nn.Sequential(
            nn.Conv3d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.Sigmoid(),
            nn.Conv3d(mid_channels, o_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.Sigmoid()
        )
        self.block3 = nn.Sequential(            
            nn.MaxPool3d(3, stride=1)
        )

    def forward(self, x):
        residual = self.block1(x)
        out = self.block2(residual)
        out = residual + out.clone()
        return self.block3(out)


class ResDecBlocks(nn.Module):
    """Class for implementing main block of decoding procedure to map extracted features to maps.

    Args:
        n_channels (int): channel size of input data
        o_channels (int): channel size of output data
    """
    def __init__(self, n_channels, o_channels):
        super(ResDecBlocks, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose3d(n_channels, o_channels, kernel_size=3),
            nn.BatchNorm3d(o_channels),
            nn.Sigmoid(),
            nn.ConvTranspose3d(o_channels, o_channels, kernel_size=3),
            nn.BatchNorm3d(o_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.block(x)


class ResEncBlocksPP(nn.Module):
    """BPARC++ : Class for implementing main block of encoding procedure to map input data to lower represeantation.

        Args:
            n_channels (int): channel size of input data
            o_channels (int): channel size of output data
            mid_channels (object:int, optional): channel size of hidden layers. Defaults to None.    

    """

    def __init__(self, n_channels, o_channels, mid_channels=None):
        super(ResEncBlocksPP, self).__init__()
        if not mid_channels:
            mid_channels = o_channels
        self.block1 = nn.Sequential(
            nn.Conv3d(n_channels, mid_channels, kernel_size=3),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU6()
        )
        self.block2 = nn.Sequential(
            nn.Conv3d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU6(),
            nn.Conv3d(mid_channels, o_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU6()
        )
        self.block3 = nn.Sequential(            
            nn.MaxPool3d(3, stride=1)
        )

    def forward(self, x):
        residual = self.block1(x)
        out = self.block2(residual)
        out = residual + out.clone()
        return self.block3(out)


class ResDecBlocksPP(nn.Module):
    """BPARC++ : Class for implementing main block of decoding procedure to map extracted features to maps.

    Args:
        n_channels (int): channel size of input data
        o_channels (int): channel size of output data
    """
    def __init__(self, n_channels, o_channels):
        super(ResDecBlocksPP, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose3d(n_channels, o_channels, kernel_size=3),
            nn.BatchNorm3d(o_channels),
            nn.ReLU6(),
            nn.ConvTranspose3d(o_channels, o_channels, kernel_size=3),
            nn.BatchNorm3d(o_channels),
            nn.ReLU6()
        )

    def forward(self, x):
        return self.block(x)