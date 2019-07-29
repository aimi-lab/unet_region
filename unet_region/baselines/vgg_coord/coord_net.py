import torchvision.models as models
import torch


class CoordNet(torch.nn.Module):
    
    def __init__(self,
                 dim_out=8,
                 checkpoint_path=None,
                 cuda=True):

        super(CoordNet, self).__init__()
        self.model = models.vgg.vgg16(pretrained=True)

        # take features and remove last maxpool layer
        self.feat_extr = self.model.features[:-1]
        self.lin = torch.nn.Linear(512 * 16 * 16, dim_out)
        self.sig = torch.nn.Sigmoid()

        self.device = torch.device('cuda' if cuda
                                    else 'cpu')

    def forward(self, x):
        x = self.feat_extr(x)
        x = self.lin(x.view(x.shape[0], -1))
        x = self.sig(x)
        return x
