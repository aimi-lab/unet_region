from VOC2012 import VOC2012
import torch
import os

# root_path = '/home/ubelix/artorg/lejeune/data'
root_path = '/home/ubelix/artorg/lejeune/data/VOCdevkit'

voc2012 = VOC2012(root_path, load_h5=False)

voc2012.read_all_data_and_save(train_data_save_path=os.path.join(root_path, 'voc2012_train.h5'),
                               val_data_save_path=os.path.join(root_path, 'voc2012_val.h5'))
voc2012.read_aug_images_labels_and_save()
