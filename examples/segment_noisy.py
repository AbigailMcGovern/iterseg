import numpy as np
import zarr
import os
from pathlib import Path
import napari
from iterseg.segmentation import affinity_unet_watershed, dog_blob_watershed



def segment_with_plateseg(image_dir, save_dir, percentages=(0.5, 1, 2, 4, 8, 16, 32)):
    segment_my_data(image_dir, save_dir, percentages, affinity_unet_watershed, 'PS')


def segment_with_DoG(image_dir, save_dir, percentages=(0.5, 1, 2, 4, 8, 16, 32)):
    segment_my_data(image_dir, save_dir, percentages, dog_blob_watershed, 'DoG')


def segment_my_data(image_dir, save_dir, percentages, func, name):
    image_dirs, save_dirs = prepare_dir_lists(image_dir, save_dir, percentages)
    v = napari.Viewer()
    for id, sd, p in zip(image_dirs, save_dirs, percentages):
        for f in os.listdir(id):
            im_p = os.path.join(id, f)
            img = np.array(zarr.open(im_p))
            v.add_image(img, scale=(4, 1, 1))
            sn = Path(im_p).stem + f'_{name}'
            func(v, v.layers['img'], sd, name=sn, debug=True)
            del v.layers['img']


def prepare_dir_lists(image_dir, save_dir, percentages):
    image_dirs = [os.path.join(image_dir, f'{p}%') for p in percentages]
    save_dirs = [os.path.join(save_dir, f'{p}%') for p in percentages]
    for sd in save_dirs:
        os.makedirs(sd, exist_ok=True)
    return image_dirs, save_dirs


image_dir = '/Users/abigailmcgovern/Data/iterseg/invivo_platelets/images'
save_dir_DoG = '/Users/abigailmcgovern/Data/iterseg/invivo_platelets/DoG'
save_dir_PS = '/Users/abigailmcgovern/Data/iterseg/invivo_platelets/segmentations_plateseg'

#segment_with_plateseg(image_dir, save_dir_PS)
segment_with_DoG(image_dir, save_dir_DoG)