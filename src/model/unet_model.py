""" Full assembly of the parts to form the complete network """

from src.model.unet_parts import *




class RUNetSmall(nn.Module):
    def __init__(self, n_channels, bilinear=False, use_tanh=False):
        super(RUNetSmall, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64, use_tanh=use_tanh))
        self.down1 = (Down(64, 128, use_tanh=use_tanh))
        
        factor = 2 if bilinear else 1
        self.down2 = (Down(128, 256 // factor, use_tanh=use_tanh))
        # self.down3 = (Down(256, 512 // factor, use_tanh=use_tanh))
        
        # self.down4 = (Down(512, 1024 // factor, use_tanh=use_tanh))
        # self.up1 = (Up(1024, 512 // factor, bilinear, use_tanh=use_tanh))
        # self.up2 = (Up(512, 256 // factor, bilinear, use_tanh=use_tanh))
        self.up3 = (Up(256, 128 // factor, bilinear, use_tanh=use_tanh))
        self.up4 = (Up(128, 64, bilinear, use_tanh=use_tanh))
        self.outc = (OutConv(64, 3))
        self.init_weights()
    def init_weights(self):
        def weight_init(m):
            if isinstance(m, nn.Conv2d):
                nn.init.constant_(m.weight, 0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.apply(weight_init)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        # x4 = self.down3(x3)

        # x5 = self.down4(x4)
        # xx = self.up1(x5, x4)
        # xx = self.up2(x4, x3)
        # xx = self.up2(xx, x3)

        xx = self.up3(x3, x2)
        xx = self.up4(xx, x1)
        resid = self.outc(xx)
        resid = F.tanh(resid)
        return x[:,-3:] + resid

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        # self.down3 = torch.utils.checkpoint(self.down3)
        # self.down4 = torch.utils.checkpoint(self.down4)
        # self.up1 = torch.utils.checkpoint(self.up1)
        # self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)


class RUNet(nn.Module):
    def __init__(self, n_channels, bilinear=False, use_tanh=False):
        super(RUNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64, use_tanh=use_tanh))
        self.down1 = (Down(64, 128, use_tanh=use_tanh))
        self.down2 = (Down(128, 256, use_tanh=use_tanh))
        self.down3 = (Down(256, 512, use_tanh=use_tanh))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor, use_tanh=use_tanh))
        self.up1 = (Up(1024, 512 // factor, bilinear, use_tanh=use_tanh))
        self.up2 = (Up(512, 256 // factor, bilinear, use_tanh=use_tanh))
        self.up3 = (Up(256, 128 // factor, bilinear, use_tanh=use_tanh))
        self.up4 = (Up(128, 64, bilinear, use_tanh=use_tanh))
        self.outc = (OutConv(64, 3))
        # self.init_weights()
    def init_weights(self):
        def weight_init(m):
            if isinstance(m, nn.Conv2d):
                nn.init.constant_(m.weight, 0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.apply(weight_init)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        xx = self.up1(x5, x4)
        xx = self.up2(xx, x3)
        xx = self.up3(xx, x2)
        xx = self.up4(xx, x1)
        resid = self.outc(xx)
        resid = F.tanh(resid)
        return x[:,-3:] + resid

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)


class UNet(nn.Module):
    def __init__(self, n_channels, bilinear=False, use_tanh=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64, use_tanh=use_tanh))
        self.down1 = (Down(64, 128, use_tanh=use_tanh))
        self.down2 = (Down(128, 256, use_tanh=use_tanh))
        self.down3 = (Down(256, 512, use_tanh=use_tanh))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor, use_tanh=use_tanh))
        self.up1 = (Up(1024, 512 // factor, bilinear, use_tanh=use_tanh))
        self.up2 = (Up(512, 256 // factor, bilinear, use_tanh=use_tanh))
        self.up3 = (Up(256, 128 // factor, bilinear, use_tanh=use_tanh))
        self.up4 = (Up(128, 64, bilinear, use_tanh=use_tanh))
        self.outc = (OutConv(64, 3))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        logits = F.sigmoid(logits)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
if __name__ == "__main__":
    big = UNet(30)
    def num_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(num_params(big))
    small = RUNetSmall(30)
    print(num_params(small))
