from dai_imports import*

def conv(in_channels,out_channels):
    m = [nn.Conv2d(in_channels,out_channels,kernel_size=5,padding=2),nn.PReLU()]
    if in_channels == out_channels:
        m.append(nn.MaxPool2d(2))
    return nn.Sequential(*m)

class LeNetPlus(nn.Module):
    def __init__(self,num_classes,feature_dim,shape):
        super(LeNetPlus, self).__init__()
        channel_pairs = [(1,32),(32,32),(32,64),(64,64),(64,128),(128,128)]
        back = [conv(*channels) for channels in channel_pairs]
        self.mult = (shape[-2]//8*shape[-1]//8)
        self.back_bone = nn.Sequential(*back)
        self.prelu = nn.PReLU()
        self.fc1 = nn.Linear(128*self.mult, feature_dim)
        self.fc2 = nn.Linear(feature_dim, num_classes, bias=False)

    def forward(self, x):
        x = self.back_bone(x)
        x = x.view(-1, 128*self.mult)
        features = self.prelu(self.fc1(x))
        classes = self.fc2(features)
        return features,classes