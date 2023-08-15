import numpy as np
import zarr
import os
import napari
from iterseg._dock_widgets import _assess_segmentation
from tifffile import imread
from pathlib import Path


def prepare_dir_lists(root_dir, percentages):
    model_dirs = [os.path.join(root_dir, f'{p}%') for p in percentages]
    model_dirs = model_dirs + [os.path.join(root_dir, 'no_noise')]
    return model_dirs


def prepare_gt_seg_dict(root_dir, gt_dir,  percentages):
    model_dirs = [os.path.join(root_dir, f'{p}%') for p in percentages]
    model_dirs = model_dirs + [os.path.join(root_dir, 'no_noise')]
    gts = [os.path.join(gt_dir, f) for f in os.listdir(gt_dir) if check_correct_ext(f)]
    gts.sort()
    md_dict = {}
    for md in model_dirs:
        md_dict[md] = {
            'files': [os.path.join(md, f) for f in os.listdir(md) if check_correct_ext(f)],
            'gts' : gts
        }
        md_dict[md]['files'].sort()
    return md_dict


def read_correct(sp):
    if sp.endswith('tif'):
        img = imread(sp)
    elif sp.endswith('zarr'):
        img = zarr.open(sp)
        img = np.array(img)
    return img


def check_correct_ext(f):
    val = f.endswith('.tif') or f.endswith('.zarr')
    return val


def assess_my_segmentations(data_dir, gt_dir, save_dir, name, percentages=(0.5, 1, 2, 4, 8, 16, 32)):
    md_dict = prepare_gt_seg_dict(data_dir, gt_dir,  percentages)
    for key in md_dict:
        segs = [read_correct(f) for f in md_dict[key]['files']]
        segs = [np.squeeze(a) for a in segs]
        segs = np.stack(segs)
        gt = [read_correct(f) for f in md_dict[key]['gts']]
        gt = np.stack(gt)
        n = Path(key).stem 
        if n == '0':
            n = '0.5%'
        if n == 'no_noise':
            n = '0%'
        n = name + '_' + n
        _assess_segmentation(gt, segs, save_dir=save_dir, name=n, show=False)


data_dir_DoG = '/Users/abigailmcgovern/Data/iterseg/invivo_platelets/DoG'
data_dir_PS = '/Users/abigailmcgovern/Data/iterseg/invivo_platelets/segmentations_plateseg'
gt_dir = '/Users/abigailmcgovern/Data/iterseg/invivo_platelets/ground_truths/people/Pia'
save_dir = '/Users/abigailmcgovern/Data/iterseg/invivo_platelets/DoG_plateseg_comp/noise_series'

assess_my_segmentations(data_dir_DoG, gt_dir, save_dir, name='DoG')
assess_my_segmentations(data_dir_PS, gt_dir, save_dir, name='PS')

