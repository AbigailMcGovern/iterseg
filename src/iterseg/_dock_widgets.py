import ast
from typing import Optional, Dict, Union

import numpy as np
import napari
from napari.qt import thread_worker
from napari_plugin_engine import napari_hook_implementation
from magicgui import widgets, magic_factory
import toolz as tz

from .predict import predict_output_chunks
from . import watershed as ws
from .training_experiments import get_experiment_dict, run_experiment
import zarr
from skimage.io import imread
import os
import pathlib

import dask.array as da
from .metrics import get_accuracy_metrics, plot_accuracy_metrics

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
    output_dir='.'
    ):
    assert image_4D_stack.shape == labels_4D_stack.shape
    channels_list = construct_channels_list(affinities_extent, mask_prediction, 
                                        centre_prediciton)
    condition_name = [training_name, ]
    image_list = [image_4D_stack[i, ...] for i in range(image_4D_stack.shape[0])]
    labels_list = [labels_4D_stack[i, ...] for i in range(labels_4D_stack.shape[0])]
    conditions_list = construct_conditions_list(image_list, loss_function, learning_rate, epochs, scale)
    exp_dict = get_experiment_dict(channels_list, condition_name, 
                                   conditions_list=conditions_list, 
                                   validation_prop=validation_prop, 
                                   n_each=n_each)
    run_experiment(exp_dict, image_list, labels_list, output_dir)


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


# ---------------
# Load train data
# ---------------

@magic_factory(
    images_path={'widget_type': 'FileEdit'}, 
    labels_path={'widget_type': 'FileEdit'}, 
    scale={'widget_type' : 'LiteralEvalLineEdit'}, 
    type={'widget_type' : 'LineEdit'}
)
def load_train_data(
    napari_viewer: napari.viewer.Viewer, 
    images_path: str, 
    labels_path: str,
    scale: tuple,
    type: str = 'Training'
    ):
    if not os.path.isdir(images_path):
        images_paths = [images_path, ]
    else:
        images_paths = [os.path.join(images_path, f) for f in os.listdir(images_path)]
    if not os.path.isdir(labels_path):
        labels_paths = [labels_path, ]
    else:
        labels_paths = [os.path.join(labels_path, f) for f in os.listdir(labels_path)]
    # check that there are the same number of files in each path
    assert len(images_paths) == len(labels_paths)
    # go though every image and read in
    img0 = read_with_correct_modality(images_paths[0]) # probs need to use tensorstore
    im_shape = img0.shape
    del img0
    lab0 = read_with_correct_modality(labels_paths[0])
    lb_shape = lab0.shape
    del lab0
    assert im_shape == lb_shape
    assert len(im_shape) == 3
    image_stack = generate_4D_stack(images_paths, im_shape)
    labels_stack = generate_4D_stack(labels_paths, lb_shape)
    napari_viewer.add_image(image_stack, scale=scale, name=f'{type} Images')
    napari_viewer.layers['Training Images'].metadata.update({'images_paths' : images_paths, 'labels_paths' : labels_paths})
    napari_viewer.add_labels(labels_stack, scale=scale, name=f'{type} Ground Truth')
    napari_viewer.layers['Training Ground Truth'].metadata.update({'images_paths' : images_paths, 'labels_paths' : labels_paths})


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
        im = zarr.open(path)
    return im



# -------------
# Load New Data
# -------------

# use mode='d' for directories, see:
# https://napari.org/magicgui/usage/_autosummary/magicgui.widgets.FileEdit.html
@magic_factory( image_path={'mode': 'd'}, labels_dir={'mode': 'd'})
def initiate_training_data(image_path: pathlib.Path, labels_dir: pathlib.Path):
    # load a single image into 4D array and initiate a labels layer (in file as tesnorstore)
    pass


@magic_factory(
    images_path={'widget_type': 'FileEdit'}, 
    max_number={'widget_type' : 'LiteralEvalLineEdit'}
)
def add_image_data(images_path, max_number=None):
    pass



# -------------------
# Segment with Model
# -------------------



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
        chunk_size: str = '(10, 256, 256)',
        margin: str = '(1, 64, 64)',
        unet: str = 'default', 
        use_default_unet: bool = True,
        num_pred_channels: int = 5,  # can probs get this from last unet layer
        ):
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
        data.shape, chunks=(1,) + data.shape[1:], dtype=np.int32
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
            name='watershed',
            scale=scale,
            translate=translate,
            )
    # thread_worker()


def predict_segment_loop():
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
        watershed_worker = create_watershed_worker(
            output_volume[slicing],
            affinities_channels=(0, 1, 2),
            thresholding_channel=3,
            centroids_channel=4,
            out=current_output.ravel(),
        )

        # define the PREDICTION WORKER
        launch_prediction_worker = thread_worker(
                predict_output_chunks,
                connect={'returned': watershed_worker.start},
                )
        # start PREDICTION WORKER, which *should* start the WATERSHED WORKER once returned
        worker = launch_prediction_worker(
                unet, input_volume, chunk_size, output_volume, margin=margin, use_default_unet=use_default_unet
                )
        ws.segment_output_image(output_volume[slicing],
            affinities_channels=(0, 1, 2),
            thresholding_channel=3,
            centroids_channel=4,
            out=current_output.ravel(),)
        output_volume[:] = 0
        yield

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


class UNetPredictWidget(widgets.Container):
    def __init__(self, napari_viewer):
        super().__init__(labels=False)
        self.predict_widget = widgets.FunctionGui(
                predict_output_chunks_widget,
                param_options=dict(
                        napari_viewer={'visible': False},
                        chunk_size={'widget_type': 'LiteralEvalLineEdit'},
                        unet={'widget_type': 'FileEdit'}, 
                        use_default_unet={'widget_type': 'CheckBox'} 
                        )
                )
        self.append(widgets.Label(value='U-net prediction'))
        self.append(self.predict_widget)
        self.predict_widget.napari_viewer.bind(napari_viewer)
        self.viewer = napari_viewer
        self.call_watershed = None


# -----------------------
# Segmentation Assessment
# -----------------------

@magic_factory(
    save_dir={'widget_type': 'FileEdit'}
)
def assess_segmentation(
    napari_viewer: napari.Viewer,
    ground_truth: napari.types.LabelsData, 
    model_segmentation: napari.types.LabelsData, 
    variation_of_information: bool, 
    average_precision: bool, 
    object_count: bool, 
    #diagnostics: bool,
    save_dir: str, 
    save_prefix: str,
    show: bool
    ):
    # save info
    os.makedirs(save_dir, exist_ok=True)
    data_path = os.path.join(save_dir, save_prefix + '_metrics.csv')
    # need to get the slices from the model-produced layer
    ms_layer = find_matching_labels(napari_viewer, model_segmentation)
    slices = ms_layer.metadata.get('slices')
    data = get_accuracy_metrics(slices, ground_truth, model_segmentation, 
                              variation_of_information, average_precision, 
                              object_count, data_path)
    # generate plots
    plot_accuracy_metrics(data, save_prefix, save_dir, show)

    # Diagnostic plots
    # plot_diagnostics(...)
    #TODO:
    # - get image metrics relating to quality/information/randomness/edges
    # - get object metrics for segmentation 


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


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    # you can return either a single widget, or a sequence of widgets
    return [UNetPredictWidget, copy_data, train_from_viewer, load_train_data]
