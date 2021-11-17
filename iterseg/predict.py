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

from . import unet
from . import watershed as ws

u_state_fn = os.path.join(
        os.path.dirname(__file__), 'data/unet-210913-zyxmc.pt'
        )

u = unet.UNet(in_channels=1, out_channels=5)
IGNORE_CUDA = False
map_location = torch.device('cpu')  # for loading the pre-existing unet
if torch.cuda.is_available() and not IGNORE_CUDA:
    u.cuda()
    map_location = None
u.load_state_dict(torch.load(u_state_fn, map_location=map_location))


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


@tz.curry
def throttle_function(func, every_n=1000):
    """Return a copy of function that only runs every n calls.

    This is useful when attaching a slow callback to a frequent event.

    Parameters
    ----------
    func : callable
        The input function.
    every_n : int
        Number of ignored calls before letting another call through.
    """
    counter = 0

    def throttled(*args, **kwargs):
        nonlocal counter
        result = None
        if counter % every_n == 0:
            result = func(*args, **kwargs)
        counter += 1
        return result

    return throttled


def predict_output_chunks(
        unet,
        input_volume,
        chunk_size,
        output_volume,
        margin=0,
        ):
    u = unet
    ndim = len(chunk_size)
    chunk_starts, chunk_crops = make_chunks(
            input_volume.shape[-ndim:], chunk_size, margin=margin
            )
    for start, crop in tqdm(list(zip(chunk_starts, chunk_crops))):
        sl = tuple(
                slice(start0, start0 + step)
                for start0, step in zip(start, chunk_size)
                )
        tensor = torch.from_numpy(input_volume[sl][np.newaxis, np.newaxis])
        if torch.cuda.is_available() and not IGNORE_CUDA:
            tensor = tensor.cuda()
        predicted_array = u(tensor).detach().cpu().numpy()
        # add slice(None) for the 5 channels
        cr = (slice(None),) + tuple(slice(i, j) for i, j in crop)
        output_volume[(slice(None),) + sl][cr] = predicted_array[(0,) + cr]
        yield
    return output_volume


if __name__ == '__main__':
    import nd2_dask as nd2
    #data_fn = '/data/platelets/200519_IVMTR69_Inj4_dmso_exp3.nd2'
    data_fn = os.path.expanduser(
            '~/Dropbox/share-files/200519_IVMTR69_Inj4_dmso_exp3.nd2'
            )
    layer_list = nd2.nd2_reader.nd2_reader(data_fn)

    t_idx = 114

    source_vol = layer_list[2][0]
    vol2predict = rescale_intensity(np.asarray(source_vol[t_idx])).astype(
            np.float32
            )
    prediction_output = np.zeros((5,) + vol2predict.shape, dtype=np.float32)

    size = (10, 256, 256)
    chunk_starts, chunk_crops = make_chunks(vol2predict.shape, size, (1, 0, 0))

    viewer = napari.Viewer(ndisplay=3)
    l0 = viewer._add_layer_from_data(*layer_list[0])[0]
    l1 = viewer._add_layer_from_data(*layer_list[1])[0]
    l2 = viewer._add_layer_from_data(*layer_list[2])[0]

    offsets = -0.5 * np.asarray(l0.scale)[-3:] * np.eye(5, 3)
    prediction_layers = viewer.add_image(
            prediction_output,
            channel_axis=0,
            name=['z-aff', 'y-aff', 'x-aff', 'mask', 'centroids'],
            scale=l0.scale[-3:],
            translate=list(np.asarray(l0.translate[-3:]) + offsets),
            colormap=[
                    'bop purple', 'bop orange', 'bop orange', 'gray', 'gray'
                    ],
            visible=[False, False, False, True, False],
            )
    viewer.dims.set_point(0, t_idx)

    def refresh_prediction_layers():
        for layer in prediction_layers:
            layer.refresh()

    labels = np.pad(
            np.zeros(prediction_output.shape[1:], dtype=np.uint32),
            1,
            mode='constant',
            constant_values=0,
            )
    labels_layer = viewer.add_labels(
            labels[1:-1, 1:-1, 1:-1],
            name='watershed',
            scale=prediction_layers[-1].scale,
            translate=prediction_layers[-1].translate,
            )

    # closure to connect to threadworker signal
    def segment(prediction):
        yield from ws.segment_output_image(
                prediction,
                affinities_channels=(0, 1, 2),
                centroids_channel=4,
                thresholding_channel=3,
                out=labels.ravel()
                )

    refresh_labels = throttle_function(labels_layer.refresh, every_n=10000)
    segment_worker = thread_worker(
            segment, connect={'yielded': refresh_labels}
            )

    prediction_worker = thread_worker(
            predict_output_chunks,
            connect={
                    'yielded': refresh_prediction_layers,
                    'returned': segment_worker,
                    },  # yapf: disable
            )
    prediction_worker(u, vol2predict, size, prediction_output, margin=0)

    napari.run()
