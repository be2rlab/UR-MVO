from math import pi

import cv2
from imgaug import augmenters as iaa
import kornia
import numpy as np
from numpy.random import uniform
from scipy.stats import truncnorm
import torch
import torch.nn.functional as f


class PhotoAugmentor:
    def __init__(self, config):
        self.config = config

    def additive_gaussian_noise(self, image):
        stddev_range = self.config['params']['additive_gaussian_noise']['stddev_range']
        stddev = np.random.uniform(stddev_range[0], stddev_range[1])
        noise = np.random.normal(scale=stddev, size=image.shape)
        noisy_image = np.clip(image+noise, 0, 255)
        return noisy_image

    def additive_speckle_noise(self, image):
        prob_range = self.config['params']['additive_speckle_noise']['prob_range']
        prob = np.random.uniform(prob_range[0], prob_range[1])
        sample = np.random.uniform(size=image.shape)
        noisy_image = np.where(sample <= prob, np.zeros_like(image), image)
        noisy_image = np.where(sample >= (1. - prob), 255.*np.ones_like(image), noisy_image)
        noisy_image = np.clip(noisy_image.round(), 0, 255)
        return noisy_image

    def random_brightness(self, image):
        brightness_max_change = self.config['params']['random_brightness']['max_abs_change']
        delta = np.random.uniform(low=-brightness_max_change, high=brightness_max_change, size=1)[0]
        image = image + delta
        image = np.clip(image, 0, 255.0)
        return image.astype(np.float32)

    def random_contrast(self, image):
        contrast_factors = tuple(self.config['params']['random_contrast']['strength_range'])
        contrast_factor = np.random.uniform(low=contrast_factors[0],
                                            high=contrast_factors[1],
                                            size=1)[0]
        mean = image.mean()
        image = (image-mean)*contrast_factor+mean
        image = np.clip(image, 0, 255.)
        return image.astype(np.float32)

    def additive_shade(self, image):
        nb_ellipses = self.config['params']['additive_shade']['nb_ellipses']
        transparency_range = self.config['params']['additive_shade']['transparency_range']
        kernel_size_range = self.config['params']['additive_shade']['kernel_size_range']

        def _py_additive_shade(img):
            min_dim = min(img.shape[:2]) / 4
            mask = np.zeros(img.shape[:2], np.uint8)
            for i in range(nb_ellipses):
                ax = int(max(np.random.rand() * min_dim, min_dim / 5))
                ay = int(max(np.random.rand() * min_dim, min_dim / 5))
                max_rad = max(ax, ay)
                x = np.random.randint(max_rad, img.shape[1] - max_rad)  # center
                y = np.random.randint(max_rad, img.shape[0] - max_rad)
                angle = np.random.rand() * 90
                cv2.ellipse(mask, (x, y), (ax, ay), angle, 0, 360, 255, -1)

            transparency = np.random.uniform(*transparency_range)
            kernel_size = np.random.randint(*kernel_size_range)
            if (kernel_size % 2) == 0:  # kernel_size has to be odd
                kernel_size += 1
            mask = cv2.GaussianBlur(mask.astype(np.float32), (kernel_size, kernel_size), 0)
            shaded = img * (1 - transparency * mask/255.)
            return np.clip(shaded, 0, 255)

        # shaded = tf.py_func(_py_additive_shade, [image], tf.float32)
        # res = tf.reshape(shaded, tf.shape(image))
        shaded = _py_additive_shade(image)
        res = np.reshape(shaded, image.shape)

        return np.clip(res.round(), 0, 255)

    def __call__(self, image):

        primitives = self.config['primitives']
        indices = np.arange(len(primitives))
        np.random.shuffle(indices)

        for i in indices:
            image = getattr(self, primitives[i])(image)

        return image.astype(np.float32)


def erosion2d(image, strel, origin=(0, 0), border_value=1e6):
    image_pad = f.pad(image, [origin[0], strel.shape[1]-origin[0]-1, origin[1], strel.shape[2]-origin[1]-1], mode='constant', value=border_value)
    image_unfold = f.unfold(image_pad, kernel_size=strel.shape[1])  # [B,C*sH*sW,L],L is the number of patches
    strel_flatten = torch.flatten(strel, start_dim=1).unsqueeze(-1)
    diff = image_unfold - strel_flatten
    # Take maximum over the neighborhood
    result, _ = diff.min(dim=1)
    # Reshape the image to recover initial shape
    return torch.reshape(result, image.shape)


def homographic_aug_pipline(img, device='cpu'):
    if len(img.shape) == 2:
        img = img.unsqueeze(dim=0).unsqueeze(dim=0)
    image_shape = img.shape[2:]  # HW
    homography = sample_homography(image_shape, device=device)
    warped_image = kornia.geometry.warp_perspective(img, homography, image_shape, align_corners=True)
    warped_valid_mask = compute_valid_mask(image_shape, homography, 0, device=device)
    return warped_image, warped_valid_mask, homography


def compute_valid_mask(image_shape, homographies, erosion_radius=0, device='cpu'):
    if len(homographies.shape) == 2:
        homographies = homographies.unsqueeze(0)
    # TODO:uncomment this line if your want to get same result as tf version
    # homographies = torch.linalg.inv(homographies)
    B = homographies.shape[0]
    img_one = torch.ones(tuple([B, 1, *image_shape]), device=device, dtype=torch.float32)  # B,C,H,W
    mask = kornia.geometry.warp_perspective(img_one, homographies, tuple(image_shape), align_corners=True)
    mask = mask.round()  # B1HW
    # mask = cv2.warpPerspective(np.ones(image_shape), homography, dsize=tuple(image_shape[::-1]))#dsize=tuple([w,h])
    if erosion_radius > 0:
        # TODO: validation & debug
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_radius*2,)*2)
        kernel = torch.as_tensor(kernel[np.newaxis, :, :], device=device)
        _, kH, kW = kernel.shape
        origin = ((kH-1)//2, (kW-1)//2)
        mask = erosion2d(mask, torch.flip(kernel, dims=[1, 2]), origin=origin) + 1.  # flip kernel so perform as tf.nn.erosion2d

    return mask.squeeze(dim=1)  # BHW


def sample_homography(shape, device='cpu'):

    config = {'perspective': True, 'scaling': True, 'rotation': True, 'translation': True,
              'n_scales': 5, 'n_angles': 25, 'scaling_amplitude': 0.2, 'perspective_amplitude_x': 0.1,
              'perspective_amplitude_y': 0.1, 'patch_ratio': 0.5, 'max_angle': pi / 2,
              'allow_artifacts': False, 'translation_overflow': 0.}
    std_trunc = 2
    # Corners of the input patch
    margin = (1 - config['patch_ratio']) / 2
    pts1 = margin + np.array([[0, 0],
                              [0, config['patch_ratio']],
                              [config['patch_ratio'], config['patch_ratio']],
                              [config['patch_ratio'], 0]])
    pts2 = pts1.copy()

    # Random perspective and affine perturbations
    if config['perspective']:
        if not config['allow_artifacts']:
            perspective_amplitude_x = min(config['perspective_amplitude_x'], margin)
            perspective_amplitude_y = min(config['perspective_amplitude_y'], margin)
        else:
            perspective_amplitude_x = config['perspective_amplitude_x']
            perspective_amplitude_y = config['perspective_amplitude_y']
        perspective_displacement = truncnorm(-std_trunc, std_trunc, loc=0., scale=perspective_amplitude_y/2).rvs(1)
        h_displacement_left = truncnorm(-std_trunc, std_trunc, loc=0., scale=perspective_amplitude_x/2).rvs(1)
        h_displacement_right = truncnorm(-std_trunc, std_trunc, loc=0., scale=perspective_amplitude_x/2).rvs(1)
        pts2 += np.array([[h_displacement_left, perspective_displacement],
                          [h_displacement_left, -perspective_displacement],
                          [h_displacement_right, perspective_displacement],
                          [h_displacement_right, -perspective_displacement]]).squeeze()

    # Random scaling
    # sample several scales, check collision with borders, randomly pick a valid one
    if config['scaling']:
        scales = truncnorm(-std_trunc, std_trunc, loc=1, scale=config['scaling_amplitude']/2).rvs(config['n_scales'])
        # scales = np.random.uniform(0.8, 2, config['n_scales'])
        scales = np.concatenate((np.array([1]), scales), axis=0)

        center = np.mean(pts2, axis=0, keepdims=True)
        scaled = (pts2 - center)[np.newaxis, :, :] * scales[:, np.newaxis, np.newaxis] + center
        if config['allow_artifacts']:
            valid = np.arange(config['n_scales'])  # all scales are valid except scale=1
        else:
            valid = (scaled >= 0.) * (scaled < 1.)
            valid = valid.prod(axis=1).prod(axis=1)
            valid = np.where(valid)[0]
        idx = valid[np.random.randint(valid.shape[0], size=1)].squeeze().astype(int)
        pts2 = scaled[idx, :, :]

    # Random translation
    if config['translation']:
        t_min, t_max = np.min(pts2, axis=0), np.min(1 - pts2, axis=0)
        if config['allow_artifacts']:
            t_min += config['translation_overflow']
            t_max += config['translation_overflow']
        pts2 += np.array([uniform(-t_min[0], t_max[0], 1), uniform(-t_min[1], t_max[1], 1)]).T

    # Random rotation
    # sample several rotations, check collision with borders, randomly pick a valid one
    if config['rotation']:
        angles = np.linspace(-config['max_angle'], config['max_angle'], num=config['n_angles'])
        angles = np.concatenate((np.array([0.]), angles), axis=0)  # in case no rotation is valid
        center = np.mean(pts2, axis=0, keepdims=True)
        rot_mat = np.reshape(np.stack([np.cos(angles), -np.sin(angles), np.sin(angles),
                                       np.cos(angles)], axis=1), [-1, 2, 2])
        rotated = np.matmul((pts2 - center)[np.newaxis, :, :], rot_mat) + center

        if config['allow_artifacts']:
            valid = np.arange(config['n_angles'])  # all scales are valid except scale=1
        else:
            valid = (rotated >= 0.) * (rotated < 1.)
            valid = valid.prod(axis=1).prod(axis=1)
            valid = np.where(valid)[0]
        idx = valid[np.random.randint(valid.shape[0], size=1)].squeeze().astype(int)
        pts2 = rotated[idx, :, :]

    # Rescale to actual size
    shape = np.array(shape[::-1])  # different convention [y, x]
    pts1 *= shape[np.newaxis, :]
    pts2 *= shape[np.newaxis, :]

    homography = cv2.getPerspectiveTransform(np.float32(pts1), np.float32(pts2))
    homography = torch.tensor(homography, device=device, dtype=torch.float32).unsqueeze(dim=0)
    homography = torch.inverse(homography)  # inverse here to be consistent with tf version
    return homography  # [1,3,3]


def ratio_preserving_resize(img, target_size):
    '''
    :param img: raw img
    :param target_size: (h,w)
    :return:
    '''
    scales = np.array((target_size[0]/img.shape[0], target_size[1]/img.shape[1]))  # h_s,w_s

    new_size = np.round(np.array(img.shape)*np.max(scales)).astype(int)
    temp_img = cv2.resize(img, tuple(new_size[::-1]))
    curr_h, curr_w = temp_img.shape
    target_h, target_w = target_size
    ##
    hp = (target_h-curr_h)//2
    wp = (target_w-curr_w)//2
    aug = iaa.Sequential([iaa.CropAndPad(px=(hp, wp, target_h-curr_h-hp, target_w-curr_w-wp), keep_size=False),])
    new_img = aug(images=temp_img)
    return new_img
