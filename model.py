import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, noise_dim : int = 100, input_chanels: int = 3) -> None:
        super(Generator, self).__init__()
        
        self.upsample_layer1 = nn.Sequential(
            # [B, 100, 1, 1] -> [B, 1024, 4, 4]
            nn.ConvTranspose2d(noise_dim, 1024, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )
        self.upsample_layer2 = nn.Sequential(
            # [B, 1024, 4, 4] -> [B, 512, 8, 8]
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        ) 
        self.upsample_layer3 = nn.Sequential(
            # [B, 512, 8, 8] -> [B, 256, 16, 16]
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        ) 
        self.upsample_layer4 = nn.Sequential(
            # [B, 256, 16, 16] -> [B, 128, 32, 32]
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.upsample_layer5 = nn.Sequential(
            # [B, 128, 32, 32] -> [B, 3, 64, 64]
            nn.ConvTranspose2d(128, input_chanels, kernel_size=4, stride=2, padding=1, bias=True),
            nn.Tanh(),
        )
        self.init_weights()

    def forward(self, x):
        x = self.upsample_layer1(x)
        x = self.upsample_layer2(x)
        x = self.upsample_layer3(x)
        x = self.upsample_layer4(x)
        x = self.upsample_layer5(x)
        return x
    
    def init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
                nn.init.constant_(m.bias.data, 0)


class Discriminator(nn.Module):
    def __init__(self, input_channels: int = 3) -> None:
        super(Discriminator, self).__init__()
        
        self.downsample_layer1 = nn.Sequential(
            # [B, 3, 64, 64]  -> [B, 64, 32, 32] 
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2),
        )
        self.downsample_layer2 = nn.Sequential(
            # [B, 64, 32, 32] -> [B, 128, 16, 16] 
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.downsample_layer3 = nn.Sequential(
            # [B, 128, 16, 16] -> [B, 256, 8, 8]
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )
        self.downsample_layer4 = nn.Sequential(
            # [B, 256, 8, 8] -> [B, 512, 4, 4] 
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
        )
        self.downsample_layer5 = nn.Sequential(
            # [B, 512, 4, 4] -> [B, 1, 1, 1]
            nn.Conv2d(512, 1, kernel_size=4, stride=2, padding=0, bias=True),
            nn.Sigmoid(),
        )
        self.init_weights()

    def forward(self, x):
        x = self.downsample_layer1(x)
        x = self.downsample_layer2(x)
        x = self.downsample_layer3(x)
        x = self.downsample_layer4(x)
        x = self.downsample_layer5(x)
        return x

    def init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
                nn.init.constant_(m.bias.data, 0)
