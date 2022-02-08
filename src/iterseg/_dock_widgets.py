import ast
from email.mime import base
from sys import prefix
from typing import Optional, Dict, Union

import numpy as np
import napari
from napari.qt import thread_worker
#from napari_plugin_engine import napari_hook_implementation
from magicgui import widgets, magic_factory
import toolz as tz
from torch import save

from .predict import predict_output_chunks, make_chunks
from . import watershed as ws
from .training_experiments import get_experiment_dict, run_experiment
import zarr
from skimage.io import imread
import os
import pathlib

import dask.array as da
from .metrics import get_accuracy_metrics, plot_accuracy_metrics

import tensorstore as ts

# ------------
# Train widget
# ------------


@magic_factory(
    call_button=True, 
    mask_prediction={'choices': ['mask', 'centreness']}, 
    centre_prediciton={'choices': ['centreness-log', 'centreness', 'centroid-gauss']},
    affinities_extent={'widget_type' : 'LiteralEvalLineEdit'},
    training_name={'widget_type': 'LineEdit'}, 
    loss_function={'choices': ['BCELoss', 'DiceLoss']}, 
    output_dir={'widget_type': 'FileEdit'}, 
    scale={'widget_type' : 'LiteralEvalLineEdit'},
    )
def train_from_viewer(
    viewer: napari.viewer.Viewer, 
    image_4D_stack: napari.types.ImageData, 
    labels_4D_stack: napari.types.LabelsData,
    scale, 
    mask_prediction='mask', 
    centre_prediciton='centreness-log', #lol btw this is a typo in the whole repo :P
    affinities_extent=1, 
    training_name='my-unet',
    loss_function='BCELoss', 
    learning_rate=0.01, 
    epochs=4,
    validation_prop=0.2, 
    n_each=50,
    output_dir='.', 
    save_labels=True,
    ):
    assert image_4D_stack.shape == labels_4D_stack.shape
    channels_list = construct_channels_list(affinities_extent, mask_prediction, 
                                        centre_prediciton)
    condition_name = [training_name, ]
    image_list = [image_4D_stack[i, ...] for i in range(image_4D_stack.shape[0])]
    labels_list = [labels_4D_stack[i, ...] for i in range(labels_4D_stack.shape[0])]
    conditions_list = construct_conditions_list(image_list, loss_function, 
                                                learning_rate, epochs, scale)
    exp_dict = get_experiment_dict(channels_list, condition_name, 
                                   conditions_list=conditions_list, 
                                   validation_prop=validation_prop, 
                                   n_each=n_each)
    u_path = run_experiment(exp_dict, image_list, labels_list, output_dir)
    if save_labels:
        save_path = os.path.join(output_dir, training_name + '_labels-prediction.zarr')
    else:
        save_path = None
    labels_layer = predict_output_chunks_widget(viewer, image_4D_stack, unet=u_path[0], 
                                                use_default_unet=False, save_path=save_path, 
                                                name=training_name)
    meta = {
        'unet' : u_path[0], 
        'chunk_size' : (10, 256, 256), 
        'margin' : (1, 64, 64), 
        'mask_prediction' : mask_prediction, 
        'centre_prediction' : centre_prediciton, 
        'affinities_extent' : affinities_extent, 
        'loss_function' : loss_function, 
        'output_dir' : output_dir, 
        'learning_rate' : learning_rate, 
        'epochs' : epochs, 
        'validation_prop' : validation_prop, 
        'n_each' : n_each, 
        'labels_path' : save_path
    }
    labels_layer.metadata.update()


def construct_channels_list(
    affinities_extent, 
    mask_prediction, 
    centre_predicition
    ):
    dims = ('z', 'y', 'x')
    affs = []
    if isinstance(affinities_extent, tuple):
        m = f'please ensure the length of the affinities extent tuple matches the number of dims in {dims}'
        assert len(affinities_extent) == len(dims), m
    elif isinstance(affinities_extent, int):
        affinities_extent = [affinities_extent, ] * len(dims)
        affinities_extent = tuple(affinities_extent)
    else:
        m = 'Please insert affinities extent of type tuple or int (e.g., 1 or (2, 2, 1))'
        raise TypeError(m)
    for i, d in enumerate(dims):
        n_affs = affinities_extent[i]
        for n in range(1, n_affs + 1):
            affs.append(f'{d}-{n}')
    affs.append(mask_prediction)
    affs.append(centre_predicition)
    affs = [tuple(affs), ]
    return affs


def construct_conditions_list(
    image_list, 
    loss_function, 
    learning_rate, 
    epochs, 
    scale
    ):
    scale = [scale for l in image_list]
    condition_dict = {
        'scale' : scale, 
        'lr' : learning_rate, 
        'loss_function' : loss_function, 
        'epochs' : epochs
    }
    return [condition_dict, ]


# -------------------------
# Load Image or Labels Data
# -------------------------

@magic_factory(
    data_path={'widget_type': 'FileEdit'}, 
    data_type={'choices': ['individual frames', 'image stacks']},
    layer_name={'widget_type' : 'LineEdit'},
    layer_type={'choices': ['Image', 'Labels']},
    scale={'widget_type' : 'LiteralEvalLineEdit'}, 
)
def load_data(
    napari_viewer: napari.viewer.Viewer, 
    data_path: str, 
    # name_pattern: str, 
    data_type: str,
    layer_name:str,
    layer_type: str,
    scale: tuple =(1, 1, 1),
    translate: tuple =(0, 0, 0),
    ):

    _load_data(napari_viewer, data_path, data_type, 
                layer_name, layer_type, scale, translate)


def _load_data(
    napari_viewer: napari.viewer.Viewer, 
    data_path: str, 
    # name_pattern: str, 
    data_type: str,
    layer_name:str,
    layer_type: str,
    scale: tuple =(1, 1, 1),
    translate: tuple =(0, 0, 0),
    ):

    if not os.path.isdir(data_path):
        data_paths = [data_path, ]
    else:
        data_paths = os.listdir(data_path)
    imgs = []
    for p in data_paths:
        im = read_with_correct_modality(p)
        imgs.append(im)
    # if data is in stack/s (4D stacks of 3D frames)
    if data_type == 'image stacks':
        imgs = np.concatenate(imgs)
    # if data is in individual 3D frames
    if data_type == 'individual frames':
        imgs = np.stack(imgs)
    if layer_type == 'Image':
        napari_viewer.add_image(imgs, scale=(1, ) + scale, name=layer_name, translate=(0,) + translate)
    if layer_type == 'Labels':
        napari_viewer.add_labels(imgs, scale= (1, ) + scale, name=layer_name, translate=(0,) + translate)


def generate_4D_stack(path_list, shape):
    images = []
    for p in path_list:
        im  = read_with_correct_modality(p)
        assert shape == im.shape
        images.append(im)
    images = np.stack(images)
    assert len(images.shape) == 4
    return images


def generate_4D_stack(path_list, shape):
    images = []
    for p in path_list:
        im  = read_with_correct_modality(p)
        assert shape == im.shape
        images.append(im)
    images = np.stack(images)
    assert len(images.shape) == 4
    return images


def read_with_correct_modality(path):
    if path.endswith('.tif') or path.endswith('.tiff'):
        im = imread(path)
    elif path.endswith('.zar') or path.endswith('.zarr'):
        im = zarr.creation.open_array(path, 'r')
    return im



# -------------------
# Predict dock widget
# -------------------

@tz.curry
def self_destructing_callback(callback, disconnect):
    def run_once_callback(*args, **kwargs):
        result = callback(*args, **kwargs)
        disconnect(run_once_callback)
        return result

    return run_once_callback


# magicfactory, forget the container below
def predict_output_chunks_widget(
        napari_viewer,
        input_volume_layer: napari.layers.Image,
        labels_layer: napari.layers.Labels, 
        chunk_size: str = '(10, 256, 256)',
        margin: str = '(1, 64, 64)',
        which_unet: str = 'default',
        unet: str = 'to choose file select above: file', 
        num_pred_channels: int = 5,  # can probs get this from last unet layer
        save_path: Union[str, None] = None,
        name: str = 'labels-prediction'
        ):
    use_default_unet = which_unet == 'default'
    if which_unet == 'file':
        unet = unet
    elif which_unet == 'labels layer':
        unet = labels_layer.metadata['unet']
        chunk_size = labels_layer.metadata['chunk_size']
        margin = labels_layer.metadata['margin']
    if type(chunk_size) is str:
        chunk_size = ast.literal_eval(chunk_size)
    if type(margin) is str:
        margin = ast.literal_eval(margin)
    viewer = napari_viewer
    data = input_volume_layer.data
    scale = viewer.layers[0].scale[1:] # lazy assumption that all layers have the same scale 
    translate = viewer.layers[0].translate[1:] # and same translate 
    print(type(data), data.shape)
    ndim = len(chunk_size)
    # For now: use in-memory zarr array. When working, we can open on-disk
    # array with tensorstore so that we can paint into it even as the network
    # is writing to other timepoints. (exploding head emoji)
    output_labels = zarr.zeros(
        data.shape, 
        chunks=(1,) + data.shape[1:], 
        dtype=np.int32, 
        )
    # in the future, we can add neural net output as an in-memory zarr array
    # that only displays the currently predicted output timepoint, and zeroes
    # out the rest. This is because the output volumes are otherwise
    # extremely large.
    output_volume = np.zeros((num_pred_channels,) + data.shape[1:], dtype=np.float32) 
    # Best would be to use napari.util.progress to have nested progress bars
    # here (one for timepoints, one for chunks, maybe one for watershed)
    output_layer = viewer.add_labels(
            output_labels,
            name=name,
            scale=scale,
            translate=translate,
            )

    def handle_yields(yielded_val):
        print(f"Completed timepoint {yielded_val}")

    #chunks = make_chunks(data[0, ...].shape, chunk_size, margin)
    #n_chunks = len(chunks[0])

    launch_worker = thread_worker(
        predict_segment_loop,
        progress={'total': data.shape[0], 'desc': 'thread-progress'},
        # this does not preclude us from connecting other functions to any of the
        # worker signals (including `yielded`)
        connect={'yielded': handle_yields},
    )

    worker = launch_worker(
        data, 
        viewer, 
        output_volume, 
        unet, 
        chunk_size, 
        margin, 
        use_default_unet,
        ndim, 
        output_labels
    )
    if save_path is not None:
        zarr.save(save_path, output_labels)

    return output_layer


def predict_segment_loop(
        data, 
        viewer, 
        output_volume, 
        unet, 
        chunk_size, 
        margin, 
        use_default_unet,
        ndim, 
        output_labels
    ):
    for t in range(data.shape[0]):
        print(t)
        viewer.dims.current_step = (t, 0, 0, 0)
        slicing = (t, slice(None), slice(None), slice(None))
        input_volume = np.asarray(data[slicing]).astype(np.float32)
        input_volume /= np.max(input_volume)
        current_output = np.pad(
            np.zeros(output_volume.shape[1:], dtype=np.uint32),
            1,
            mode='constant',
            constant_values=0,
            )
        crop = tuple([slice(1, -1),] * ndim)  # yapf: disable
        # predict using unet
        predict_output_chunks(
            unet, 
            input_volume, 
            chunk_size, 
            output_volume, 
            margin=margin, 
            use_default_unet=use_default_unet
            )
        ws.segment_output_image(
            output_volume, #[slicing],
            affinities_channels=(0, 1, 2),
            thresholding_channel=3,
            centroids_channel=4,
            out=current_output.ravel())
        output_labels[t, ...] = current_output[crop]
        output_volume[:] = 0
        yield t


# ------------------------------------------------
# What was copy_data for... might need to ask Juan
@magic_factory
def copy_data(
        napari_viewer: napari.viewer.Viewer,
        source_layer: napari.layers.Layer,
        target_layer: napari.layers.Layer,
        ):
    src_data = source_layer.data
    dst_data = target_layer.data

    ndim_src = src_data.ndim
    ndim_dst = dst_data.ndim
    slice_ = napari_viewer.dims.current_step
    slicing = slice_[:ndim_dst - ndim_src]
    dst_data[slicing] = src_data
# ------------------------------------------------


class UNetPredictWidget(widgets.Container):
    def __init__(self, napari_viewer):
        super().__init__(labels=False)
        self.predict_widget = widgets.FunctionGui(
                predict_output_chunks_widget,
                param_options=dict(
                        napari_viewer={'visible': False},
                        chunk_size={'widget_type': 'LiteralEvalLineEdit'},
                        unet={'widget_type': 'FileEdit'}, 
                        which_unet={'choices': ['default', 'file', 'labels layer']}
                        )
                )
        self.append(widgets.Label(value='U-net prediction'))
        self.append(self.predict_widget)
        self.predict_widget.napari_viewer.bind(napari_viewer)
        self.viewer = napari_viewer
        self.call_watershed = None


# -----------------------
# Ground Truth Generation
# -----------------------

@magic_factory()
def generate_ground_truth(
    napari_veiwer: napari.Viewer,
    labels_to_correct: napari.layers.Labels, 
    new_layer_name: str = 'Ground truth', 
    save_path: str= './ground_truth.zarr'
):  
    chunks = (1, ) + tuple(labels_to_correct.data.shape[1:])
    labels = zarr.zeros_like(labels_to_correct.data, chunks=chunks)
    labels[:] = labels_to_correct.data
    zarr.save(save_path, labels)
    spec = {
         'driver': 'zarr',
         'kvstore': {
             'driver': 'file',
             'path': save_path,
         },
         'metadata' : {
             'dataType': str(labels.dtype),
             'dimensions': list(labels.shape),
             'blockSize': list(chunks),
         },
         'create': False,
         'delete_existing': False,
     }
    labels = ts.open(spec)
    napari_veiwer.add_labels(
        labels, 
        name=new_layer_name, 
        scale=labels_to_correct.scale, 
        translate=labels_to_correct.translate)


# --------------
# Combine Layers
# --------------

@magic_factory()
def combine_layers(
    napari_viewer: napari.Viewer, 
    base_layer: napari.layers.Layer, 
    to_append: napari.layers.Layer, 
    save_dir: Union[str, None] = None, 
    save_prefix: str = '', 
    save_all: bool = True, # save all or just new
    save_indivdually: bool =  False, 
    number_from: int = 0
):
    '''
    Combine a stack of labels volumes with a second stack. 

    Parameters
    ----------
    napari_viewer: napari.Viewer
    base_layer: napari.layers.Layer
        Layer to which the second layer will be added.
    to_append: napari.layers.Labels
        Layer to append to the base layer.
    save_dir: str or None
        If not none, the output will be saved to this 
        directory.
    save_prefix: str
        If save_dir is not None, this is the name that will 
        be used in saving the images.
    save_all: bool
        If save_dir is not None, should all of the images in 
        the stack be saved? If False, only the appended images
        will be saved
    save_individually: bool
        Should images be saved as a series of volumes or a single 
        4D stack?
    number_from: int
        if save_individually, images will be saved each with a unique 
        number, starting with this one (e.g., range starting from 0)
    '''
    base_layer.data = np.concatenate([base_layer.data, to_append.data])
    # what to do about metadata
    # should probably assert that the layer properties match

    # saving stuff... yay
    if save_dir is not None:
        if not save_all:
            if not save_indivdually:
                save_path = os.path.join(save_dir, save_prefix + '.zarr')
                zarr.save(save_path, to_append.data)
            else:
                for t in range(to_append.data.shape[0]):
                    save_path = os.path.join(save_dir, 
                                        save_prefix + f'_{t + number_from}.zarr')
                    zarr.save(save_path, to_append.data[t, ...])
        else:
            if not save_indivdually:
                save_path = os.path.join(save_dir, save_prefix + '.zarr')
                zarr.save(save_path, base_layer.data)
            else:
                for t in range(base_layer.data.shape[0]):
                    save_path = os.path.join(save_dir, 
                                        save_prefix + f'_{t + number_from}.zarr')
                    zarr.save(save_path, base_layer.data[t, ...])



# -----------------------
# Segmentation Assessment
# -----------------------

@magic_factory(
    save_dir={'widget_type': 'FileEdit'}
)
def assess_segmentation(
    napari_viewer: napari.Viewer,
    ground_truth: napari.layers.Image, 
    model_segmentation: napari.layers.Labels, 
    chunk_size: tuple = (10, 256, 256), 
    margin: tuple = (1, 64, 64),
    variation_of_information: bool = True, 
    average_precision: bool = True, 
    object_count: bool = True, 
    #diagnostics: bool,
    save_dir: str = './', 
    save_prefix: str = 'segmentation-metrics',
    show: bool = True
    ):
    # save info
    os.makedirs(save_dir, exist_ok=True)
    data_path = os.path.join(save_dir, save_prefix + '_metrics.csv')
    # need to get the slices from the model-produced layer
    meta = model_segmentation.metadata
    chunk_size, margin = chunk_n_margin(meta, chunk_size, margin)
    shape = model_segmentation.data.shape
    slices = get_slices_from_chunks(shape, chunk_size, margin)
    data = get_accuracy_metrics(slices, ground_truth.data, model_segmentation.data, 
                                variation_of_information, average_precision, 
                                object_count, data_path)
    # generate plots
    plot_accuracy_metrics(data, save_prefix, save_dir, show)

    # Diagnostic plots
    # plot_diagnostics(...)
    #TODO:
    # - get image metrics relating to quality/information/randomness/edges
    # - get object metrics for segmentation 


def chunk_n_margin(meta, chunk_size, margin):
    mcs = meta.get('chunk_size')
    if mcs is not None:
        chunk_size = mcs
    mmg = meta.get('margin')
    if mmg is not None:
        margin = mmg
    return chunk_size, margin


def get_slices_from_chunks(arr_shape, chunk_size, margin):
    chunk_starts, chunk_crops = make_chunks(arr_shape, chunk_size, margin)
    if len(arr_shape) <= 3:
        ts = range(1)
    else:
        ts = range(arr_shape[0])
    slices = []
    for t in ts:
        for start, crop in list(zip(chunk_starts, chunk_crops)):
            sl = (slice(t, t+1), ) + tuple(
                    slice(start0, start0 + step)
                    for start0, step in zip(start, chunk_size)
                    )
            cr = tuple(slice(i, j) for i, j in crop)
            slices.append((sl, cr)) # useage: 4d_labels[sl][cr]
    return slices

# -------------------
# Validation Analysis
# -------------------

@magic_factory()
def validation_analysis(
    images: napari.types.ImageData,
    model_1: napari.types.LabelsData,
    model_2: napari.types.LabelsData,
    model_3: napari.types.LabelsData,
    model_4: napari.types.LabelsData, 
    ):
    # want to compute quality metrics etc for every chunk of data
    # passed through the pipeline 
    # therefore need the unet chunks info
    # probs get input chunk info from unet state dict
    pass


# ----------------
# Helper Functions
# ----------------

def find_matching_labels(
    napari_viewer: napari.Viewer, 
    labels
    ):
    # indices for the non background labels
    lab_idxs = np.where(labels > 0)
    matches = []
    for i, l in enumerate(napari_viewer.layers):
        if isinstance(l, napari.layers.labels.labels.Labels):
            res = np.min(l.data[lab_idxs] == labels[lab_idxs])
            if res == True:
                matches.append(i)
    if len(matches) > 1:
        print('multiple identical labels found... using the first...')
    return napari_viewer.layers[matches[0]]



# -------------------
# Hook implementation
# -------------------


#@napari_hook_implementation
#def napari_experimental_provide_dock_widget():
    # you can return either a single widget, or a sequence of widgets
    #return [UNetPredictWidget, copy_data, train_from_viewer, load_train_data]
