import utils
from dai_imports import*
from pixel_shuffle import PixelShuffle_ICNR

def icnr(x, scale=2, init=nn.init.kaiming_normal_):
    new_shape = [int(x.shape[0] / (scale ** 2))] + list(x.shape[1:])
    subkernel = torch.zeros(new_shape)
    subkernel = init(subkernel)
    subkernel = subkernel.transpose(0, 1)
    subkernel = subkernel.contiguous().view(subkernel.shape[0],
                                            subkernel.shape[1], -1)
    kernel = subkernel.repeat(1, 1, scale ** 2)
    transposed_shape = [x.shape[1]] + [x.shape[0]] + list(x.shape[2:])
    kernel = kernel.contiguous().view(transposed_shape)
    kernel = kernel.transpose(0, 1)
    return kernel

def conv(ni, nf, kernel_size=3, actn=True):
    layers = [nn.Conv2d(ni, nf, kernel_size, padding=kernel_size//2)]
    if actn: layers.append(nn.ReLU(True))
    return nn.Sequential(*layers)

class ResSequential(nn.Module):
    def __init__(self, layers, res_scale=1.0):
        super().__init__()
        self.res_scale = res_scale
        self.m = nn.Sequential(*layers)

    def forward(self, x):
        x = x + self.m(x) * self.res_scale
        return x

def res_block(nf, res_scale=0.1):
    return ResSequential([conv(nf, nf), conv(nf, nf, actn=False)], res_scale)

def upsample(ni, nf, scale):
    layers = []
    for _ in range(int(math.log(scale,2))):
        layer = [conv(ni, nf*4), nn.PixelShuffle(2)]
        kernel = icnr(layer[0][0].weight, scale=scale)
        layer[0][0].weight.data.copy_(kernel)
        layers += layer
    return nn.Sequential(*layers)


class SrResnet(nn.Module):
    def __init__(self, scale=2, res_blocks=8, res_channels=32, res_scale=0.1, shuffle_blur=True):
        super().__init__()
        features = [conv(3, res_channels)]
        for _ in range(res_blocks): features.append(res_block(res_channels, res_scale))
        # shuffle = nn.Sequential(*([PixelShuffle_ICNR(res_channels, res_channels, 2, shuffle_blur)] * (int(math.log(scale,2)))))
        shuffle = PixelShuffle_ICNR(res_channels, res_channels, scale, shuffle_blur)
        features += [conv(res_channels,res_channels), shuffle,
                     nn.BatchNorm2d(res_channels),
                     conv(res_channels, 3, actn=False)]
        # features += [PixelShuffle_ICNR(res_channels, res_channels, 2, shuffle_blur), nn.Conv2d(res_channels, 3, 3, 2)]
        # features += [conv(res_channels,res_channels), upsample(res_channels, res_channels, scale),
        #              nn.BatchNorm2d(res_channels),
        #              conv(res_channels, 3, actn=False)]
        self.features = nn.Sequential(*features)        
    def forward(self, x): return self.features(x)