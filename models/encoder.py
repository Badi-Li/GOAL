"""
Defining convolutional encoder for partial semantic map.
We fix the encoder to be two residual blocks and two downsample blocks, downsampling
the map by a factor of 4.
"""
import torch.nn as nn 
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out

class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, latent_channels):
        super().__init__()
        self.res1 = ResidualBlock(in_channels)
        self.down1 = Downsample(in_channels, latent_channels)
        self.res2 = ResidualBlock(latent_channels)
        self.down2 = Downsample(latent_channels, out_channels)

    def forward(self, x):
        x = self.res1(x)
        x = self.down1(x)
        x = self.res2(x)
        x = self.down2(x)
        return x