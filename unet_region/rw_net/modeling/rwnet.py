from unet_region.rw_net.modeling.deeplab import DeepLab
from torch import nn
import torch

class PairwiseSimilarity(nn.Module):
    def __init__(self, radius_rel=0.02):
        super(PairwiseSimilarity, self).__init__()

    def forward(self, x):
        radius = self.radius_rel * x.shape[-1]

        batch_size, chans, w, h = x.shape
        y = [x_[c] for c in range(chans) for x_ in x]

        
    

class RWNet(nn.Module):
    def __init__(self,
                 cp_path=None):
        super(RWNet, self).__init__()

        self.deeplab_model = DeepLab(backbone='drn')

        if(cp_path is not None):
            cp = torch.load(
                cp_path, map_location='cpu')
            self.deeplab_model.load_state_dict(cp['state_dict'])

        # predict two classes (background / foreground)
        self.deeplab_model.decoder.last_conv[-1] = nn.Conv2d(256, 2, kernel_size=1, stride=1)

    def forward(self, x):
        
