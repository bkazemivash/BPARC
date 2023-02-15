"""Base UNet style model with encoding and decoding blocks with modifications."""

from torch import nn 

class BaseUnetModel(nn.Module):
    """Implementation of BPARC-UNet model for brain parcellation.

    Args:
        in_dim (int): size of input layer
        hidden_dim (int): size of hidden layer
        kernel (int, optional): size of kernel. Defaults to 3.
        use_drop (bool, optional): using drop out layer. Defaults to False.
        drop_ratio (float, optional): drop out ratio. Defaults to .2.
    """  
    def __init__(self, kernel, use_drop=False, drop_ratio=.2):      
        super(BaseUnetModel, self).__init__()
        self.stage1 = StemUnit(1, 8, kernel=kernel)
        self.stage2 = ResEncBlocks(8, 16, kernel=kernel)
        self.stage3 = ResEncBlocks(16, 32, kernel=kernel)
        self.stage4 = ResEncBlocks(32, 64, kernel=kernel)
        if use_drop:
            self.stage5 = nn.Dropout3d(p=drop_ratio)
        self.stage6 = ResDecBlocks(64, 32, kernel=kernel)
        self.stage7 = ResDecBlocks(32, 16, kernel=kernel)
        if use_drop:
            self.stage8 = nn.Dropout3d(p=drop_ratio)
        self.stage9 = ResDecBlocks(16, 8, kernel=kernel)
        self.stage10 = FinalUnit(8,1, kernel=kernel)

    def forward(self, x):
        residual1 = self.stage1(x)
        residual2 = self.stage2(residual1)
        residual3 = self.stage3(residual2)
        out = self.stage4(residual3)
        out = self.stage5(out)
        out = self.stage6(out)
        out = residual3 + out.clone()
        out = self.stage7(out)
        out = self.stage8(out)
        out = residual2 + out.clone()
        out = self.stage9(out)
        out = residual1 + out.clone()
        return self.stage10(out)

class StemUnit(nn.Module):
    def __init__(self, in_dim, hidden_dim, kernel=3):
        """Implementation of entry unit

        Args:
            in_dim (int): size of input
            hidden_dim (int): size of hidden layer
            kernel (int, optional): _description_. Defaults to 3.
        """        
        super(StemUnit, self).__init__()
        self.stage1 = nn.Sequential(
            nn.Conv3d(in_dim, hidden_dim, kernel_size=kernel, bias=False),
            nn.ReLU6()
        )        

    def forward(self, x):
        return self.stage1(x)

class FinalUnit(nn.Module):
    def __init__(self, in_dim, hidden_dim, kernel=3):
        """Implementation of final unit

        Args:
            in_dim (int): size of input
            hidden_dim (int): size of hidden layer
            kernel (int, optional): size of kernel. Defaults to 3.
        """
        super(FinalUnit, self).__init__()
        self.stage1 = nn.Sequential(
            nn.ConvTranspose3d(in_dim, hidden_dim, kernel_size=kernel, bias=False),
            nn.Sigmoid()
        )        

    def forward(self, x):
        return self.stage1(x)

class ResEncBlocks(nn.Module):
    """Implementation of encoding (down) module, to be used in BPARC unet model.

    Args:
        in_dim (int): size of input layer
        hidden_dim (int): size of hidden layer
        kernel (int, optional): size of kernel. Defaults to 3.
    """    
    def __init__(self, in_dim, hidden_dim, kernel=3):    
        super(ResEncBlocks, self).__init__()
        self.stage1 = nn.Sequential(
            nn.Conv3d(in_dim, hidden_dim, kernel_size=kernel, bias=False),
            nn.BatchNorm3d(hidden_dim),
            nn.Sigmoid()
        )
        self.stage2 = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=kernel, padding=1, bias=False),
            nn.BatchNorm3d(hidden_dim),
            nn.Sigmoid(),
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=kernel, padding=1, bias=False),
            nn.BatchNorm3d(hidden_dim),
            nn.Sigmoid()
        )
        self.stage3 = nn.Sequential( 
            nn.Sigmoid(),           
            nn.MaxPool3d(3, stride=1)
        )

    def forward(self, x):
        residual = self.stage1(x)
        out = self.stage2(residual)
        out = residual + out.clone()
        return self.stage3(out)


class ResDecBlocks(nn.Module):
    """Implementation of decoding (up) module, to be used in BPARC unet model.

    Args:
        in_dim (int): channel size of input layer
        hidden_dim (int): channel size of hidden layer
        kernel (int, optional): size of kernel. Defaults to 3.
    """
    def __init__(self, in_dim, hidden_dim, kernel=3):    
        super(ResDecBlocks, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose3d(in_dim, hidden_dim, kernel_size=kernel, bias=False),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU6(),
            nn.ConvTranspose3d(hidden_dim, hidden_dim, kernel_size=kernel, bias=False),
            nn.BatchNorm3d(hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.block(x)