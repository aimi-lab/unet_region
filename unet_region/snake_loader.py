from torch.utils import data

class SnakeLoader(data.Dataset):
    def __init__(self,
                 make_snake=False,
                 make_edt=False,
                 tsdf_thr=False,
                 length_snake=30):

        self.make_edt = make_edt
        self.tsdf_thr = tsdf_thr

        self.make_snake = make_snake
        self.length_snake = length_snake

        self.to_collate_keys = ['image',
                                'label/segmentation']


        if(tsdf_thr is not None):
            self.to_collate_keys.append('label/tsdf')
            self.to_collate_keys.append('phi_tilda')
            self.to_collate_keys.append('label/U')

        if(make_edt is not None):
            self.to_collate_keys.append('label/edt_D')
            self.to_collate_keys.append('label/edt_beta')
            self.to_collate_keys.append('label/edt_kappa')

    def __getitem__(self, index):

        sample = super().__getitem__(index)
