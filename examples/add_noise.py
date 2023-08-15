import numpy as np
import zarr
import os
from tifffile import imread
from pathlib import Path


def add_noise_func(img, percent):
    noise = percent / 100
    noise_vol = np.random.random(img.shape) * noise * img.max()
    img = img + noise_vol
    return img


def read_correct(sp):
    if sp.endswith('tif'):
        img = imread(sp)
    elif sp.endswith('zarr'):
        img = zarr.open(sp)
        img = np.array(img)
    return img


def save_im(img, d, p, ip):
    sn = Path(ip).stem + f'_{p}%.zarr'
    sp = os.path.join(d, sn)
    zarr.save(sp, img)


def save_noisy_images(im_dir, root_dir, percentages=(0.5, 1, 2, 4, 8, 16, 32)):
    dir_paths = [os.path.join(root_dir, f'{p}%') for p in percentages]
    img_paths = [os.path.join(im_dir, f) for f in os.listdir(im_dir) if f.endswith('.tif') or f.endswith('.zarr')]
    for p, d in zip(percentages, dir_paths):
        os.makedirs(d, exist_ok=True)
        for ip in img_paths:
            img = read_correct(ip)
            img = add_noise_func(img, p)
            save_im(img, d, p, ip)

im_dir = '/Users/abigailmcgovern/Data/iterseg/invivo_platelets/images/no_noise'
root_dir = '/Users/abigailmcgovern/Data/iterseg/invivo_platelets/images'
save_noisy_images(im_dir, root_dir)
