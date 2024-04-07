import os
import numpy as np
import glob
from copy import deepcopy
from SP.augmentation import homographic_aug_pipline
from SP.augmentation import PhotoAugmentor
import torch
import cv2


class AQUADataset(torch.utils.data.Dataset):

    def __init__(self, enable_homo = True, enable_photo= True, device='cpu'):
        super(AQUADataset, self).__init__()

        self.device = device
        self.resize = (320, 256)
        config = {
            'primitives': [
                'random_brightness', 'random_contrast', 'additive_speckle_noise',
                'additive_gaussian_noise', 'additive_shade'
            ],
            'params': {
                'random_brightness': {'max_abs_change': 50},
                'random_contrast': {'strength_range': [0.5, 1.5]},
                'additive_gaussian_noise': {'stddev_range': [0, 10]},
                'additive_speckle_noise': {'prob_range': [0, 0.0035]},
                'additive_shade': {
                    'transparency_range': [-0.5, 0.5],
                    'kernel_size_range': [100, 150],
                    'nb_ellipses': 15
                },
            }
        }

        self.photo_augmentor = PhotoAugmentor(config)
        # load config
        self.config = config  # dict_update(getattr(self, 'default_config', {}), config)
        # get images
        self.samples = self._init_data('data/imgs/train/exp/')

        self.enable_homo = enable_homo
        self.enable_photo = enable_photo

    def _init_data(self, image_path ):
        image_types = ['jpg', 'jpeg', 'bmp', 'png']
        samples = []
        for it in image_types:
            temp_im = glob.glob(os.path.join(image_path, '*.{}'.format(it)))
            temp = [{'image': imp} for imp in temp_im]
            samples += temp
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        '''load raw data'''
        data_path = self.samples[idx]  # raw image path of processed image and point path
        img = cv2.imread(data_path['image'], 0)  # Gray image
        img = cv2.resize(img, self.resize)

        # init data dict
        img_tensor = torch.as_tensor(img.copy(), dtype=torch.float, device=self.device)
        valid_mask = torch.ones(img.shape, device=self.device)

        data = {'raw': {'img': img_tensor,
                        'mask': valid_mask},
                'warp': None,
                'homography': torch.eye(3, device=self.device)}
        data['warp'] = deepcopy(data['raw'])

        ##

        if self.enable_homo:  # homographic augmentation
            # return dict{warp:{img:[H,W], point:[N,2], valid_mask:[H,W], homography: [3,3]; tensors}}
            w_image, w_mask, homography = homographic_aug_pipline(data['warp']['img'],device=self.device)
            data['warp']['img'] = w_image
            data['warp']['mask'] = w_mask
            data['homography'] = homography

        if self.enable_photo:
            photo_img = data['warp']['img'].cpu().numpy().round().astype(np.uint8)
            photo_img = self.photo_augmentor(photo_img)
            data['warp']['img'] = torch.as_tensor(photo_img, dtype=torch.float, device=self.device)

        # normalize
        data['raw']['img'] = data['raw']['img']/255.
        data['raw']['img'] = data['raw']['img'][None]
        data['warp']['img'] = data['warp']['img']/255.
        data['warp']['img'] = data['warp']['img'][0]
        data['homography'] = data['homography'][0]
        data['warp']['mask'] = data['warp']['mask'][0]

        return data 


