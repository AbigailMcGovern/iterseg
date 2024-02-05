# coding: utf-8
import os
import itertools
import torch
from tqdm import tqdm
import numpy as np
import toolz as tz
import napari
from napari.qt import thread_worker
from skimage.exposure import rescale_intensity
import platform

from . import unet as unet_mod
from . import watershed as ws

# ----------------------------------------------------------------------------
# The following was copied from https://github.com/jni/platelet-unet-watershed
# ----------------------------------------------------------------------------
IGNORE_CUDA = False

DEFAULT_UNET_PATH = os.path.join(
            os.path.dirname(__file__), 'data/232208_161159_plateseg.pt'
            )

def load_unet(u_state_fn=DEFAULT_UNET_PATH):
    if u_state_fn is None:
        u_state_fn = DEFAULT_UNET_PATH
    u = unet_mod.UNet(in_channels=1, out_channels=5)
    device = get_device()
    map_location = device  # for loading the pre-existing unet
    if torch.cuda.is_available() and not IGNORE_CUDA:
        u.cuda()
        map_location = None
    u.load_state_dict(torch.load(u_state_fn, map_location=map_location))
    return u


def make_chunks(arr_shape, chunk_shape, margin):
    ndim = len(arr_shape)
    if type(margin) == int:
        margin = [margin] * ndim
    starts = []
    crops = []
    for dim in range(ndim):
        arr = arr_shape[dim]
        chk = chunk_shape[dim]
        mrg = margin[dim]
        start = np.arange(0, arr - 2*mrg, chk - 2*mrg)
        start[-1] = arr - chk
        if len(start) > 1 and start[-1] == start[-2]:
            # remove duplicates in case last step is perfect
            start = start[:-1]
        starts.append(start)
        crop = np.array([(mrg, chk - mrg),] * len(start))  # yapf: disable
        crop[0, 0] = 0
        crop[-1, 0] = chk - (arr - np.sum(crop[:-1, 1] - crop[:-1, 0]))
        crop[-1, 1] = chk
        crops.append(crop)
    chunk_starts = list(itertools.product(*starts))
    chunk_crops = list(itertools.product(*crops))
    return chunk_starts, chunk_crops


def process_chunks(
        input_volume,
        chunk_size,
        output_volume,
        margin,
        process_data_function,
        config=None
        ):
    # kewy word arguments for the processing function
    if config is None:
        config = {}
    # get the chunks
    ndim = len(chunk_size)
    chunk_starts, chunk_crops = make_chunks(
            input_volume.shape[-ndim:], chunk_size, margin=margin
            )
    # loop to go through the chunks - make parallel if get the chance 
    for start, crop in tqdm(list(zip(chunk_starts, chunk_crops))):
        sl = tuple(
                slice(start0, start0 + step)
                for start0, step in zip(start, chunk_size)
                )
        sl = (slice(None), ) + sl
        predicted_array = process_data_function(input_volume, sl, **config)
        # check the ndims to adjust crop
        p_dim = predicted_array.ndim
        o_dim = output_volume.ndim
        cr = (slice(None),) * (p_dim - o_dim) + tuple(slice(i, j) for i, j in crop), 
        cr = cr[0]
        # get the desired crop
        pred_c = (0, ) + cr
        output_volume[sl][cr] = predicted_array[pred_c]
    return output_volume



def predict_chunk_feature_map(
        input_volume, 
        sl, 
        unet=False, 
        default_only_mask=False, 
        **kwargs
        ):
    '''
    Parameters
    ----------
    input_volume: array
        Image to be processed by the unet.
    u: unet.UNet
        The U Net you will produce the feature map with. 
    sl: tuple of slice 
        Slice to take from the input_volume. Needs to have the 
        same ndim as input_volume.
    '''
    assert unet != False, 'Please ensure a unet is loaded and supplied'
    sl = sl[1:]
    tensor = torch.from_numpy(input_volume[sl][np.newaxis, np.newaxis])
    if torch.cuda.is_available() and not IGNORE_CUDA:
        tensor = tensor.cuda()
    predicted_array = unet(tensor).detach().cpu().numpy() #
    if default_only_mask: # if using default network but only the mask channel is wanted
        predicted_array = predicted_array[3, ...]
    return predicted_array



def get_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device



# ----------------
# To Be Deprecated
# ----------------


def predict_output_chunks(
        unet, # now a string: 'path/to/unet' - only used if use_default_unet == False
        input_volume,
        chunk_size,
        output_volume,
        margin=0,
        use_default_unet=True, 
        mask_unet=False
        ):
    if unet is None:
        u = load_unet()
    else:
        u = load_unet(unet)
    #print(u)
    ndim = len(chunk_size)
    chunk_starts, chunk_crops = make_chunks(
            input_volume.shape[-ndim:], chunk_size, margin=margin
            )
    # add a for-loop for higher dims (input-volume.shape[:-ndim])
    # OR, do it one level up
    for start, crop in tqdm(list(zip(chunk_starts, chunk_crops))):
        sl = tuple(
                slice(start0, start0 + step)
                for start0, step in zip(start, chunk_size)
                )
        tensor = torch.from_numpy(input_volume[sl][np.newaxis, np.newaxis])
        if torch.cuda.is_available() and not IGNORE_CUDA:
            tensor = tensor.cuda()
        predicted_array = u(tensor).detach().cpu().numpy()
        #print(predicted_array.shape)
        # add slice(None) for the 5 channels
        cr = (slice(None),) + tuple(slice(i, j) for i, j in crop), 
        if not mask_unet:
            output_volume[(slice(None),) + sl][cr] = predicted_array[(0,) + cr]
        elif mask_unet and use_default_unet:
            # the mask is in channel 5
            output_volume[(slice(3,4), ) + sl][cr] = predicted_array[(0,) + cr]
        else:
            # assume that the output volume has only one channel
            output_volume[sl][cr] = predicted_array[(0,) + cr]
        #yield
    #print('op: ', output_volume.shape, np.max(output_volume))
    return output_volume

