from itertools import permutations
import numpy as np
from skimage.util import random_noise, img_as_float
from skimage.transform import AffineTransform, warp
from zarr.core import Array


def augment_images(image, labels, ground_truth=None, augment_prob=0.9):
    augment = np.random.binomial(1, augment_prob)
    if augment:
        image = augment_intensity(image)
    imgs = [image, ]
    if isinstance(labels, dict):
        for key in labels.keys():
            imgs.append(labels[key])
    else:
        imgs.append(labels)
    if ground_truth is not None:
        imgs.append(ground_truth)
    if augment:
        imgs = augment_order(imgs)
    result = [imgs[0], ]
    if isinstance(labels, dict):
        keys = list(labels.keys())
        labs = {key : imgs[i + 1] for i, key in enumerate(keys)}
    else:
        labs = imgs[1]
    result.append(labs)
    if ground_truth is not None:
        gt = imgs[-1]
        result.append(gt)
    return tuple(result)


def augment_intensity(
            image, 
            min_shift=-0.1, 
            max_shift=0.1, 
            min_scale=0.8, 
            max_scale=1.2, 
            shift_sigma=0.02, 
            scale_sigma=.05, 
            noise_prob=0.3, 
            verbose=False
        ):
    if isinstance(image, Array):
        image = np.array(image)
    out = image.copy() / image.max()
    # Get the intensity scale
    scale = continuous_choice(min_scale, max_scale, scale_sigma, loc=1.)
    # Get the intensity shift
    shift = continuous_choice(min_shift, max_shift, shift_sigma)
    out = (out * scale) + shift 
    # if add noise is chosen, add some type of noise to the image
    add_noise = np.random.binomial(1, noise_prob)
    if add_noise:
        options = ['gaussian', 'localvar', 'poisson', 'speckle', 'gaussian', 'speckle']
        i = np.random.randint(len(options))
        mode = options[i]
        kwargs = {}
        if mode in ['gaussian', 'speckle']:
            kwargs['var'] = 0.001
        if verbose:
            print(f'adding {mode} noise')
        out = random_noise(out, mode=mode, **kwargs)
    else:
        # clip the brightness range between 0-1. 
        # only necessary if noise isnt added because the noise function 
        # will do this for us
        out = np.where(out < 0, 0, out)
        out = np.where(out > 1, 1, out)
    return out


def augment_order(
            images, # need to take a list of images to support image and labels
            mirror_prob=0.2, 
            transpose_prob=0.2, 
            used_axes=(-2, -1), # must be in the order they appear in original image shape
            verbose=False
        ):
    out = [_copy_for_zarr(image) for image in images]
    new_out = []
    mirror = np.random.binomial(1, mirror_prob)
    if mirror:
        i = np.random.randint(0, len(used_axes))
        axis = used_axes[i]
        if verbose:
            print('mirroring along ', axis)
    for img in out:   
        s_ = [slice(None, None)] * len(img.shape)
        if mirror:
            s_[axis] = slice(None, None, -1)
        img = img[tuple(s_)]
        new_out.append(img)
    out = new_out
    # get the axis permulations for each image
    transpose = np.random.binomial(1, transpose_prob)
    if transpose:
        ps = permutations(used_axes)
        ps = [p for p in ps if p != tuple(used_axes)]
        idx = np.random.randint(0, len(ps))
        p = ps[idx]
    new_out = []
    for image in out:
        # apply permuatations randomly
        if transpose: 
            axes = [i for i in range(image.ndim)]
            for i, ax in enumerate(used_axes):
                na = p[i]
                if na < 0: # convert negative indices to positive
                    na = len(axes) + na
                axes[ax] = na
            if verbose:
                print('transposing to: ', axes)
            image = np.transpose(image, axes)
        new_out.append(image)
    return new_out


def _copy_for_zarr(image):
    if isinstance(image, Array):
        image = np.array(image) 
    return image.copy()


def augment_affine(
        image, 
        min_scale=0.9, 
        max_scale=1.1, 
        scale_sigma=0.05,
        min_rotation=None, 
        max_rotation=None, 
        rotation_sigma=None,
        min_shear=None,
        max_shear=None,
        shear_sigma=None,
        min_translation=None,
        max_translation=None, 
        translation_sigma=None, 
    ):
    """
    Need to calculate where image edges will be!!
    """
    
    scale = continuous_choice(max_scale, min_scale, scale_sigma, loc=1.)
    rotation = continuous_choice(max_rotation, min_rotation, rotation_sigma)
    shear = continuous_choice(max_shear, min_shear, shear_sigma)
    translation = continuous_choice(max_translation, min_translation, translation_sigma)
    # assumes axis 0 is z. We don't want to distort z due to anisotropy
    tform = AffineTransform(
            scale=scale, 
            rotation=rotation, 
            shear=shear, 
            translation=translation
        )
    # find the required slice of image for affine
    bs_ = ...
    # find the slice required to get correct sized data from image
    ns_ = ...
    # get the image
    out = []
    for i in range(image.shape[0]):
        plane = warp(image[i, ...], tform.inverse)
        out.append(plane)
    out = np.stack(out)



def continuous_choice(min_, max_, sigma, loc=0., size=1):
    '''
    Choose a continuous augmentation parameter from between a selected range.
    Parameter choice is approximately normally distributed (if truncated)
    with std dev 'sigma' and mean 'loc'.
    '''
    done = False
    c = 0
    while not done:
        out = np.random.normal(loc=loc, scale=sigma, size=size)
        if size == 1:
            val = out
        else:
            val = out.mean()
        done = val >= min_ and val <= max_
    return out


if __name__ == '__main__':
    import os
    import zarr
    import napari
    data_dir = '/Users/amcg0011/Data/pia-tracking/cang_training'
    img = zarr.open(os.path.join(data_dir, '191113_IVMTR26_I3_E3_t58_cang_training_image.zarr'))
    lab = zarr.open(os.path.join(data_dir, '191113_IVMTR26_I3_E3_t58_cang_training_labels.zarr'))
    v = napari.Viewer()
    for i in range(5):
        print(i)
        #nimg = augment_intensity(img)
        #nimg = augment_order([nimg])[0]
        nimg, nlab = augment_images(img, lab)
        v.add_image(nimg, visible=False)
        v.add_labels(nlab, visible=False)
    napari.run()
    

