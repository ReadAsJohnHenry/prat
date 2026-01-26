import torch
import torch.nn as nn
import torch.nn.functional as F



class DoubleConv(nn.Module):
	
	def __init__(self, in_channels, out_channels):
		super().__init__()
		# store the convolution and RELU layers
		self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 
		                kernel_size=3, padding=1, 
			            bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 
                        kernel_size=3, padding=1, 
                        bias=False),
			
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
		
	def forward(self, x):
		return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size = 2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_features_map, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_features_map = n_features_map
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        self.down4 = (Down(512, 1024))
        factor = 2 if bilinear else 1
        self.down5 = (Down(1024, 2048 // factor))
        self.up1 = (Up(2048, 1024 // factor, bilinear))
        self.up2 = (Up(1024, 512 // factor, bilinear))
        self.up3 = (Up(512, 256 // factor, bilinear))
        self.up4 = (Up(256, 128 // factor, bilinear))
        self.up5 = (Up(128, n_features_map, bilinear))
        # self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        # logits = self.outc(x)
        return x
    
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        # Gate signal from deeper layer
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        # Skip connection from encoder
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        # Fusion path
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x, return_coefficients=False):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        if return_coefficients:
            return out, psi
        return out
    
class AttentionUNet(nn.Module):
    def __init__(self, n_channels, n_features_map, bilinear=False):
        super(AttentionUNet, self).__init__()
        self.n_channels = n_channels
        self.n_features_map = n_features_map
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        factor = 2 if bilinear else 1
        self.down5 = Down(1024, 2048 // factor)

        # Attention Gates
        # self.att1 = AttentionGate(F_g=1024//factor, F_l=1024, F_int=512)
        # self.att2 = AttentionGate(F_g=512//factor, F_l=512, F_int=256)
        # self.att3 = AttentionGate(F_g=256//factor, F_l=256, F_int=128)
        # self.att4 = AttentionGate(F_g=128//factor, F_l=128, F_int=64)
        # self.att5 = AttentionGate(F_g=n_features_map, F_l=64, F_int=32)
        self.att1 = AttentionGate(F_g=1024, F_l=1024, F_int=512)
        self.att2 = AttentionGate(F_g=512, F_l=512,  F_int=256)
        self.att3 = AttentionGate(F_g=256,  F_l=256,  F_int=128)
        self.att4 = AttentionGate(F_g=128,  F_l=128,  F_int=64)
        self.att5 = AttentionGate(F_g=64,  F_l=64,   F_int=32)

        # Decoder
        self.up1 = Up(2048, 1024 // factor, bilinear)
        self.up2 = Up(1024, 512 // factor, bilinear)
        self.up3 = Up(512, 256 // factor, bilinear)
        self.up4 = Up(256, 128 // factor, bilinear)
        self.up5 = Up(128, n_features_map, bilinear)

    def forward(self, x, store_att_maps=True):
        # Encoder 
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)

        # Decoder + Attention Gates
        att_maps = []

        x = self.up1.up(x6) 
        if store_att_maps:
            x5_att, m = self.att1(g=x, x=x5, return_coefficients=True)
            att_maps.append(m)
        else:
            x5_att = self.att1(g=x, x=x5)
        x = self.up1.conv(torch.cat([x5_att, x], dim=1))

        x = self.up2.up(x)
        if store_att_maps:
            x4_att, m = self.att1(g=x, x=x4, return_coefficients=True)
            att_maps.append(m)
        else:
            x4_att = self.att2(g=x, x=x4)
        x = self.up2.conv(torch.cat([x4_att, x], dim=1))

        x = self.up3.up(x)
        if store_att_maps:
            x3_att, m = self.att1(g=x, x=x3, return_coefficients=True)
            att_maps.append(m)
        else:
            x3_att = self.att3(g=x, x=x3)
        x = self.up3.conv(torch.cat([x3_att, x], dim=1))

        x = self.up4.up(x)
        if store_att_maps:
            x2_att, m = self.att1(g=x, x=x2, return_coefficients=True)
            att_maps.append(m)
        else:
            x2_att = self.att4(g=x, x=x2)
        x = self.up4.conv(torch.cat([x2_att, x], dim=1))

        x = self.up5.up(x)
        if store_att_maps:
            x1_att, m = self.att1(g=x, x=x1, return_coefficients=True)
            att_maps.append(m)
        else:
            x1_att = self.att5(g=x, x=x1)
        x = self.up5.conv(torch.cat([x1_att, x], dim=1))

        self.last_attention_maps = att_maps

        return x