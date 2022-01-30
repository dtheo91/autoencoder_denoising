import torch
from torch import nn
import torch.nn.functional as F

class SimpleAutoencoder(nn.Module):

    """
    self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),


        )
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 64 * 12 * 12)

        self.decoder = nn.Sequential(
            torch.nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            torch.nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            torch.nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    """

    def __init__(self):
        super(SimpleAutoencoder, self).__init__()
        
        self.attention1 = CNNAttention(3, 32)
        self.inc = Conv_Layer(3, 64)

        self.attention2 = CNNAttention(64, 64)
        self.down1 = DownSampling(64, 128)

        self.attention3 = CNNAttention(128, 128)
        self.down2 = DownSampling(128, 256)

        self.attention4 = CNNAttention(256, 256)
        self.down3 = DownSampling(256, 512)

        self.attention5 = CNNAttention(512, 512)
        self.down4 = DownSampling(512, 1024)

        self.up1 = UpSampling(1024, 512)
        self.up2 = UpSampling(512, 256)
        self.up3 = UpSampling(256, 128)
        self.up4 = UpSampling(128, 64)

        self.outc = nn.Sequential(
                        nn.Conv2d(64, 3, kernel_size=1),
                        nn.Sigmoid()
                    )
        

    def forward(self, x):
        x1 = self.inc(self.attention1(x))
        x2 = self.down1(self.attention2(x1))
        x3 = self.down2(self.attention3(x2))
        x4 = self.down3(self.attention4(x3))
        x5 = self.down4(self.attention5(x4))

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        out = self.outc(x)

        return out

class Conv_Layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False), # add one more conv_layer (so triple_conv)
            #nn.Dropout2d(p=0.15),
            nn.BatchNorm2d(out_channels),
            #nn.LeakyReLU(inplace=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class DownSampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            Conv_Layer(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class UpSampling(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = Conv_Layer(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes,  out_planes, 1, bias=False)
        self.relu1 = nn.ReLU() #nn.LeakyReLU()
        self.fc2 = nn.Conv2d( out_planes, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out) * x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        return self.sigmoid(out) * x

class CNNAttention(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(CNNAttention, self).__init__()

        self.channel_att = ChannelAttention(in_planes, out_planes)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        out = self.channel_att(x)
        out = self.spatial_att(out)
        return out