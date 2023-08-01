from .augment import augment_images
from datetime import datetime
from .helpers import get_files, log_dir_or_None, write_log, LINE
import numpy as np
import os
import pandas as pd
from pathlib import Path
import re
import skimage.filters as filters
from skimage.measure import regionprops
from skimage.morphology._util import _offsets_to_raveled_neighbors
from tifffile import TiffWriter, imread
from time import time
import torch 
from tqdm import tqdm
import zarr


# -----------------------------------------------------------------------------
# Lable Generating Functions
# -----------------------------------------------------------------------------

def get_training_labels(
                        l, 
                        channels=('z-1', 'y-1', 'x-1', 'centreness'),
                        scale=(4, 1, 1)):
    labels = []
    get_offsets = False
    for chan in channels:
        if chan.startswith('offsets-'):
            get_offsets = True
    if get_offsets:
        offsets = get_centre_offsets(l, scale)
    for chan in channels:
        if chan.startswith('z'):
            axis = 0
        elif chan.startswith('y'):
            axis = 1
        elif chan.startswith('x'):
            axis = 2
        n = re.search(r'\d+', chan)
        if n is not None:
            # get the nth affinity
            n = int(n[0]) 
            lab = nth_affinity(l, n, axis)
        elif chan == 'centreness':
            # get the centreness score
            lab = get_centreness(l, scale=scale)
        elif chan == 'centreness-log':
            lab = get_centreness(l, scale=scale, log=True)
        elif chan == 'centroid-gauss':
            lab = get_gauss_centroids(l)
        elif chan.startswith('offsets-'):
            a = _offset_channel(chan)
            lab = offsets[a]
        elif chan == 'mask':
            lab = get_semantic_labels(l)
        else:
            m = f'Unrecognised channel type: {chan} \n'
            m = m + 'Please enter str of form <axis>-<n> for nth affinity (e.g., z-1), \n'
            m = m + 'centreness for centreness score (option of -log for log of centreness),\n'
            m = m + 'or offset-<axis> (e.g., offset-z) for axis offsets'
            raise ValueError(m)
        if chan.endswith('-smooth'):
            lab = smooth(lab)
        labels.append(lab)
    labels = np.stack(labels, axis=0)
    return labels


def _offset_channel(chan):
    if chan.endswith('z'):
        a = 0
    elif chan.endswith('y'):
        a = 1
    elif chan.endswith('x'):
        a = 2
    else:
        raise ValueError(f'Incompatible offset axis name: {chan}')
    return a


# ----------
# Affinities
# ----------

def nth_affinity(labels, n, axis):
    affinities = []
    labs_pad = np.pad(labels, n, mode='reflect')
    for i in range(labels.shape[axis]):
        s_0 = [slice(None, None)] * len(labs_pad.shape) 
        s_0[axis] = slice(i, i + 1)
        s_0 = tuple(s_0)
        s_n = [slice(None, None)] * len(labs_pad.shape) 
        s_n[axis] = slice(i + n, i + n + 1)
        s_n = tuple(s_n)
        new_0 = labs_pad[s_0]
        new_1 = labs_pad[s_n]
        new = new_0 - new_1
        new = np.squeeze(new)
        if len(new) > 0:
            affinities.append(new)
    affinities = np.stack(affinities, axis=axis)
    s_ = [slice(n, -n)] * len(labs_pad.shape)
    s_[axis] = slice(None, None)
    s_ = tuple(s_)
    affinities = affinities[s_]
    affinities = np.where(affinities != 0, 1., 0.)
    return affinities


# not currently referenced, uses nth_affinity() for generality
def get_affinities(image):
    """
    Get short-range voxel affinities for a segmentation. Affinities are 
    belonging to {0, 1} where 1 represents a segment boarder voxel in a
    particular direction. Affinities are produced for each dimension of 
    the labels and each dim has its own channel (e.g, (3, z, y, x)). 

    Note
    ----
    Others may represent affinities with {-1, 0}, because technically... 
    My network wasn't designed for this :)
    """
    padded = np.pad(image, 1, mode='reflect')
    affinities = []
    for i in range(len(image.shape)):
        a = np.diff(padded, axis=i)
        a = np.where(a != 0, 1.0, 0.0)
        a = a.astype(np.float32)
        s_ = [slice(1, -1)] * len(image.shape)
        s_[i] = slice(None, -1)
        s_ = tuple(s_)
        affinities.append(a[s_])
    affinities = np.stack(affinities)
    return affinities    


# -----------
# Centredness
# -----------

def get_centreness(labels, scale=(4, 1, 1), log=False, power=False):
    """
    Obtains a centreness score for each voxel belonging to a labeled object.
    Values in each object sum to one. Values are inversely proportional
    to euclidian distance from the object centroid.

    Notes
    -----
    Another possible implementation would involve the medioid, as in: 
    Lalit, M., Tomancak, P. and Jug, F., 2021. Embedding-based Instance 
    Segmentation of Microscopy Images. arXiv.

    Unfortunately, skimage doesn't yet have a method for finding the  
    medioid (more dev, *sigh*).
    """
    scale = np.array(scale)
    def dist_score(mask):
        output = np.zeros_like(mask, dtype=np.float32)
        c = np.mean(np.argwhere(mask), axis=0)
        indices, values = inverse_dist_score(
                mask, c, scale, log=log, power=power
                )
        output[indices] = values
        return output
    t = time()
    props = regionprops(labels, extra_properties=(dist_score,))
    new = np.zeros(labels.shape, dtype=np.float32)
    for i, prop in tqdm(enumerate(props), desc='Score centreness'):
        new[prop.slice] += prop.dist_score
    new = np.nan_to_num(new)
    print('------------------------------------------------------------')
    print(f'Obtained centreness scores in {time() - t} seconds')
    return new


def inverse_dist_score(mask, centroid, scale, log, power):
    '''
    Compute euclidian distances of each index from a mask
    representing a single object from the centroid of said object

    Uses scale to account for annisotropy in image
    '''
    indices = np.argwhere(mask > 0)
    distances = []
    centre = centroid
    for i in range(indices.shape[0]):
        ind = indices[i, ...]
        diff = (centre - ind) * scale
        dist = np.linalg.norm(diff)
        if log and abs(dist) > 0:
            m = f'Infinite value with distance of {dist}'
            dist = np.log(dist)
            assert not np.isinf(dist), m
        if power:
            dist = 2 ** dist
        distances.append(dist)
    distances = np.array(distances)
    if log:
        distances = distances + np.abs(distances.min()) # bring min value to 0
    norm_distances = distances / distances.max()
    values = (1 - norm_distances) 
    indices = tuple(indices.T.tolist())
    return indices, values


# --------------
# Centre Offsets
# --------------

def get_centre_offsets(labels, scale):
    m = labels > 0
    m = []
    scale = np.array(scale)
    def offsets(mask):
        shape = np.insert(mask.shape, -3, 3)
        output = np.zeros(shape, dtype=np.float32) 
        #print(output.shape)
        c = np.mean(np.argwhere(mask), axis=0)
        indices, values = centre_offsets(c, mask, scale)
        #print(indices, values)
        output[indices] = values
        return output
    t = time()
    props = regionprops(labels, extra_properties=(offsets,))
    #new = np.zeros(np.insert(labels.shape, -3, 3), dtype=np.float32)
    m = labels > 0
    m = np.stack([m, m.copy(), m.copy()], axis=0)
    new = np.where(m == 1, 0., 0.5)
    for i, prop in tqdm(enumerate(props), desc='Get axial centre offsets'):
        #print(new.shape, prop.offsets.shape, prop.slice)
        s_ = [slice(None, None), ]
        for s in prop.slice:
            s_.append(s)
        s_ = tuple(s_)
        new[s_] += prop.offsets
    new = np.nan_to_num(new)
    print('------------------------------------------------------------')
    print(f'Obtained centre offsets in {time() - t} seconds')
    return new


def centre_offsets(c, mask, scale, axes=3):
    idxs = np.argwhere(mask > 0)
    indices = []
    distances = []
    for a in range(axes):
        a_distances = []
        a_indices = []
        for i in range(idxs.shape[0]):
            idx = idxs[i, ...]
            diff = (c - idx) * scale
            # add indicies and offset
            a_indices.append(np.insert(idx, 0, a))
            a_distances.append(diff[a])
        a_indices = np.array(a_indices)
        indices.append(a_indices)
        a_distances = np.array(a_distances)
        new = []
        for d in a_distances:
            if d > 0:
                new.append((d / a_distances.max()))
            elif d == 0:
                new.append(0)
            elif d < 0:
                new.append(-(d / a_distances.min()))
        d = np.array(new)
        d = d - (-1)
        d = d / 2
        distances.append(d)
    indices = np.concatenate(indices)
    indices = tuple(indices.T.tolist())
    distances = np.concatenate(distances)
    return indices, distances



# ---------------
# Semantic Labels
# ---------------

def get_semantic_labels(labels):
    out = np.where(labels > 1, 1., 0.)
    return out


# ------------------
# Smoothed Centroids
# ------------------

# not used
def get_gauss_centroids(labels, sigma=1, z=0):
    centroids = [prop['centroid'] for prop in regionprops(labels)]
    centroids = tuple(np.round(np.stack(centroids).T).astype(int))
    centroid_image = np.zeros(labels.shape, dtype=float)
    centroid_image[centroids] = 1.
    gauss_cent = []
    for i in range(labels.shape[z]):
        s_ = [slice(None, None)] * labels.ndim
        s_[z] = slice(i, i+1)
        s_ = tuple(s_)
        plane = np.squeeze(centroid_image[s_])
        gauss_cent.append(filters.gaussian(plane, sigma=sigma))
    out = np.stack(gauss_cent, axis=z)
    out = out - out.min()
    out = out / out.max()
    #print(out.dtype, out.shape, out.max(), out.min())
    return out


def smooth(image, z=0, sigma=1):
    out = []
    for i in range(image.shape[z]):
        s_ = [slice(None, None)] * image.ndim
        s_[z] = slice(i, i+1)
        s_ = tuple(s_)
        plane = np.squeeze(image[s_])
        out.append(filters.gaussian(plane, sigma=sigma))
    out = np.stack(out, axis=z)
    return out


# -----------------------------------------------------------------------------
# Log and Print
# -----------------------------------------------------------------------------

def print_labels_info(channels, out_dir=None, log_name='log.txt'):
    print(LINE)
    if isinstance(channels, list):
        s = f'Training labels have {len(channels)} output channels: \n'
        _write_log_0(s, out_dir, log_name)
        print(s)
    if isinstance(channels, dict):
        keys = list(channels.keys())
        n_labs = len(keys)
        s = f'{n_labs} sets of training labels were generated:'
        _write_log_0(s, out_dir, log_name)
        print(s)
        for key in keys:
            s = f'Training labels entitled {key} has {len(channels[key])} output channels:'
            print(s)
            _write_log_0(s, out_dir, log_name)
            _print_chans(channels[key], out_dir, log_name)


def _write_log_0(s, out_dir, log_name):
    if out_dir is not None:
            write_log(LINE, out_dir, log_name)
            write_log(s, out_dir, log_name)


def _print_chans(channels, out_dir, log_name):
    for i, chan in enumerate(channels):
        affinity_match = re.search(r'[xyz]-\d*', chan)
        if affinity_match is not None:
            n = f'{affinity_match[0]} affinities'
        elif chan == 'centreness':
            n = 'centreness score'
        elif chan == 'centreness-log':
            n = 'log centreness score'
        elif chan == 'centroid-gauss':
            n = 'gaussian centroids'
        elif chan.startswith('offsets'):
            a = chan[-1]
            n = f'{a}-axis centre offsets'
        elif chan == 'mask':
            n = 'object mask'
        else:
            n = 'Unknown channel type'
        s = f'Channel {i}: {n}'
        print(s)
        if out_dir is not None:
            write_log(s, out_dir, log_name)