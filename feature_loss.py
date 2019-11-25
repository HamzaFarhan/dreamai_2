import utils
from dai_imports import*

def gram_matrix(x):
    n,c,h,w = x.size()
    x = x.view(n, c, -1)
    return (x @ x.transpose(1,2))/(c*h*w)

class FeatureLoss(nn.Module):
    def __init__(self, blocks_start=2, blocks_end=5, layer_wgts=[5,15,2], basic_loss=F.l1_loss, use_perceptual=True, device='cpu'):
        super().__init__()
        self.use_perceptual = use_perceptual
        self.basic_loss = basic_loss
        if use_perceptual:
            self.m_feat = vgg16_bn(True).features.to(device).eval()
            for p in self.m_feat.parameters():
                p.requires_grad = False
            blocks = [i-1 for i,o in enumerate(list(self.m_feat.children())) if isinstance(o,nn.MaxPool2d)]
            layer_ids = blocks[blocks_start:blocks_end]
            self.loss_features = [self.m_feat[i] for i in layer_ids]
            self.sfs = [utils.SaveFeatures(lf) for lf in self.loss_features]
            self.wgts = layer_wgts

    def make_features(self, x, clone=False):
        self.m_feat(x)
        return [(o.features.data.clone() if clone else o.features.data) for o in self.sfs]
    
    def forward(self, input, target):
        base_loss = self.basic_loss
        self.feat_losses = [base_loss(input,target)]
        if self.use_perceptual:
            out_feat = self.make_features(target, clone=True)
            in_feat = self.make_features(input)
            self.feat_losses += [base_loss(f_in, f_out)*w
                                for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
            self.feat_losses += [base_loss(gram_matrix(f_in), gram_matrix(f_out))*w**2 * 5e3
                                for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        return sum(self.feat_losses)