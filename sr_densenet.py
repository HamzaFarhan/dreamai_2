import utils
from dai_imports import*
from pixel_shuffle import PixelShuffle_ICNR

def conv(ni, nf, kernel_size=3, actn=True):
    layers = [nn.Conv2d(ni, nf, kernel_size, padding=kernel_size//2)]
    if actn: layers.append(nn.ReLU(True))
    return nn.Sequential(*layers)

class DenseSequential(nn.Module):
    def __init__(self, layers, dense_scale=1.0):
        super().__init__()
        self.dense_scale = dense_scale
        self.m = nn.Sequential(*layers)

    def forward(self, x):
        x = torch.cat([x,(self.m(x) * self.dense_scale)], dim=1)
        return x

def dense_block(nf, dense_scale=0.1):
    return DenseSequential([conv(nf, nf), conv(nf, nf, actn=False)], dense_scale)

class SrDensenet(nn.Module):
    def __init__(self, scale=2, dense_blocks=8, dense_channels=32, dense_scale=0.1, shuffle_blur=True):
        super().__init__()
        features = [conv(3, dense_channels)]
        for _ in range(dense_blocks): features.append(dense_block(dense_channels, dense_scale))
        # features.append(utils.Printer())
        shuffle = nn.Sequential(*([PixelShuffle_ICNR(dense_channels, dense_channels, 2, shuffle_blur)] * (int(math.log(scale,2)))))
        # shuffle = PixelShuffle_ICNR(dense_channels, dense_channels, scale, shuffle_blur)
        features += [conv(dense_channels,dense_channels), shuffle,
                     nn.BatchNorm2d(dense_channels),
                     conv(dense_channels, 3, actn=False)]
        # features.append(utils.Printer())
        # features += [conv(64,64), upsample(64, 64, scale),
        #              nn.BatchNorm2d(64),
        #              conv(64, 3, actn=False)]
        self.features = nn.Sequential(*features)        
    def forward(self, x): return self.features(x)