import ast
from typing import Union
import numpy as np
import napari
from napari.qt import thread_worker
from magicgui import widgets, magic_factory
import toolz as tz
from .predict import predict_output_chunks, make_chunks
from . import watershed as ws
from .training_experiments import get_experiment_dict, run_experiment
import zarr
from skimage.io import imread
import os
from .metrics import get_accuracy_metrics, plot_accuracy_metrics
import tensorstore as ts
import pandas as pd
from .plots import comparison_plots
from typing import Union
import json
from pathlib import Path


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
    output_dir={'widget_type': 'FileEdit', 'mode' : 'd'}, 
    scale={'widget_type' : 'LiteralEvalLineEdit'},
    )
def train_from_viewer(
    viewer: napari.viewer.Viewer, 
    image_stack: napari.layers.Image, 
    labels_stack: napari.layers.Labels,
    output_dir: Union[str, None]=None, 
    scale: tuple=(1, 1, 1), 
    mask_prediction='mask', 
    centre_prediciton='centreness-log', #lol btw this is a typo in the whole repo :P
    affinities_extent=1, 
    training_name='my-unet',
    loss_function='BCELoss', 
    learning_rate=0.01, 
    epochs=4,
    validation_prop=0.2, 
    n_each=50,
    save_labels=True,
    ):
    _train_from_viewer(viewer, image_stack, labels_stack, output_dir, scale, 
        mask_prediction, centre_prediciton, affinities_extent, training_name, 
        loss_function, learning_rate, epochs, validation_prop, n_each, save_labels)


def _train_from_viewer(
    viewer: napari.viewer.Viewer, 
    image_stack: napari.layers.Image, 
    labels_stack: napari.layers.Labels,
    output_dir: Union[str, None]=None, 
    scale: tuple=(1, 1, 1), 
    mask_prediction='mask', 
    centre_prediciton='centreness-log', #lol btw this is a typo in the whole repo :P
    affinities_extent=1, 
    training_name='my-unet',
    loss_function='BCELoss', 
    learning_rate=0.01, 
    epochs=4,
    validation_prop=0.2, 
    n_each=50,
    predict_labels: bool=True,
    save_labels: bool=True,
    ):

    # Prepare training data
    # ---------------------
    if isinstance(image_stack, napari.layers.Image):
        image_4D_stack = image_stack.data
    else:
        image_4D_stack = image_stack
    if isinstance(labels_stack, napari.layers.Labels):
        labels_4D_stack = labels_stack.data
    else:
        labels_4D_stack = labels_stack
    assert image_4D_stack.shape == labels_4D_stack.shape

    condition_name = [training_name, ]
    image_list = [image_4D_stack[i, ...] for i in range(image_4D_stack.shape[0])]
    labels_list = [labels_4D_stack[i, ...] for i in range(labels_4D_stack.shape[0])]
    del image_4D_stack
    del labels_4D_stack

    # Construct training info
    # -----------------------
    channels_list = construct_channels_list(affinities_extent, mask_prediction, 
                                        centre_prediciton)
    conditions_list = construct_conditions_list(image_list, loss_function, 
                                                learning_rate, epochs, scale)
    exp_dict = get_experiment_dict(channels_list, condition_name, 
                                   conditions_list=conditions_list, 
                                   validation_prop=validation_prop, 
                                   n_each=n_each)

    # Run training experiment
    # -----------------------
    u_path = run_experiment(exp_dict, image_list, labels_list, output_dir)

    # Predict full labels
    # -------------------
    if predict_labels:
        if save_labels:
            save_path = os.path.join(output_dir, training_name + '_labels-prediction.zarr')
        else:
            save_path = None
        labels_layer = predict_output_chunks_widget(viewer, image_stack, None, unet=u_path[0], 
                                                which_unet='file', save_path=save_path, 
                                                name=training_name)
    
    # Save metadata
    # -------------
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
    if isinstance(labels_layer, napari.layers.Labels):
        labels_layer.metadata.update(meta)
    json_object = json.dumps(meta, indent=4)
    meta_path = os.path.join(output_dir, Path(u_path).stem + '_meta.json')
    with open(meta_path, "w") as outfile:
        outfile.write(json_object)



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
    directory={'widget_type': 'FileEdit', 'mode' : 'd'}, 
    data_file={'widget_type': 'FileEdit'},
    data_type={'choices': ['individual frames', 'image stacks']},
    layer_name={'widget_type' : 'LineEdit'},
    layer_type={'choices': ['Image', 'Labels']},
    scale={'widget_type' : 'LiteralEvalLineEdit'}, 
    translate={'widget_type' : 'LiteralEvalLineEdit'}, 
)
def load_data(
    napari_viewer: napari.viewer.Viewer, 
    layer_name:str,
    layer_type: str,
    data_type: str ='individual frames',
    directory: Union[str, None] =None, 
    data_file: Union[str, None] =None,
    scale: tuple =(1, 1, 1),
    translate: tuple =(0, 0, 0),
    split_channels: bool=False
    ):
    '''
    Load the data into the viewer as a stack of 3D image frames. 

    Parameters
    ----------
    data_path: str
        Path to the image stack (if data_type is "image stacks") or to 
        the directory with image frames (if data_type is "individual frames"). 
        This needs to end in .tiff/.tif or .zarr/.zar. 
    data_type: str
        Is are data in a 4D array (e.g., 3D images are stacked together into single file)? 
        If so use "image stack". Are the data a series of individual 3D images?
        If so use "individual images".
    layer_type: str
        Is the data you want to load a segmentation (use "Labels") or an image (use "Image")
    scale: tuple of float
        Scale of the image/segmentation to display in z, y, x format
    translate: tuple of float
        Translation of the image origin in z, y, x format
    '''
    _load_data(napari_viewer, layer_name, 
               layer_type, data_type,
                 directory, data_file,
                scale, translate, split_channels)


def _load_data(
    napari_viewer: napari.viewer.Viewer, 
    layer_name: str,
    layer_type: str,
    data_type: str ='individual frames',
    directory: Union[str, None] =None, 
    data_file: Union[str, None] =None,
    scale: tuple =(1, 1, 1),
    translate: tuple =(0, 0, 0),
    split_channels: bool=False
    ):
    '''
    Load the data into the viewer as a stack of 3D image frames. 

    Parameters
    ----------
    data_path: str
        Path to the image stack (if data_type is "image stacks") or to 
        the directory with image frames (if data_type is "individual frames"). 
        This needs to end in .tiff/.tif or .zarr/.zar. 
    data_type: str
        Is are data in a 4D array (e.g., 3D images are stacked together into single file)? 
        If so use "image stack". Are the data a series of individual 3D images?
        If so use "individual images".
    layer_type: str
        Is the data you want to load a segmentation (use "Labels") or an image (use "Image")
    scale: tuple of float
        Scale of the image/segmentation to display in z, y, x format
    translate: tuple of float
        Translation of the image origin in z, y, x format
    '''
    if directory is not None:
        directory = str(directory)
    if data_file is not None:
        data_file = str(data_file)
    imgs, uses_directory = read_data(directory, data_file, data_type)
    if imgs.ndim > 3:
        if not split_channels:
            add_to_scale = (1, ) * (imgs.ndim - 3)
            add_to_translate = (0, ) * (imgs.ndim - 3)
        else: 
            add_to_scale = (1, ) * (imgs.ndim - 4)
            add_to_translate = (0, ) * (imgs.ndim - 4)
        scale = add_to_scale + scale
        translate = add_to_translate + translate
    if layer_type == 'Image':
        if not split_channels:
            napari_viewer.add_image(imgs, scale=scale, name=layer_name, translate=translate)
        else:
            for i in range(imgs.shape[0]):
                napari_viewer.add_image(imgs[i, ...], scale=scale, name=layer_name, translate=translate)
    if layer_type == 'Labels':
        napari_viewer.add_labels(imgs, scale=scale, name=layer_name, translate=translate)



def read_data(directory, data_file, data_type):
    possible_suf = ['.zarr', '.zar', '.tiff', '.tif']
    # is the data coming from a directory (not a zarr file)
    uses_directory = directory is not None
    if uses_directory:
        uses_directory = os.path.isdir(directory) and not directory.endswith('.zarr') and not directory.endswith('.zar')
    # is the data coming from a single file
    single_file = data_file is not None
    if single_file:
        if data_file.endswith('.tiff') or data_file.endswith('.tif'):
            data_paths = [data_file, ]
    elif not uses_directory:
        is_zarr = directory.endswith('.zarr') or directory.endswith('.zar')
        if is_zarr:
            data_paths = [directory, ]
    elif uses_directory:
        data_paths = []
        for f in os.listdir(directory):
            bool_list = [f.endswith(s) for s in possible_suf]
            if True in bool_list:
                data_paths.append(os.path.join(directory, f))
    imgs = []
    data_paths = sorted(data_paths)
    for p in data_paths:
        im = read_with_correct_modality(p)
        imgs.append(im)
    if uses_directory:
        # if data is in stack/s (4D stacks of 3D frames)
        if data_type == 'image stacks' and len(imgs) > 1:
            imgs = np.concatenate(imgs)
        # if data is in individual 3D frames
        if data_type == 'individual frames':
            imgs = np.stack(imgs)
    else:
        imgs = imgs[0]
    return imgs, uses_directory


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
        im = np.array(im)
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
    #print(type(data), data.shape)
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

    worker = launch_worker( # this is where the process dies... "There appear to be 2 leaked semaphore objects to clean up at shutdown"
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
        #print(t)
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
#@magic_factory(
#    data_path={'widget_type': 'FileEdit'}, 
#    data_type={'choices': ['individual frames', 'image stacks']},
#    layer_name={'widget_type' : 'LineEdit'},
#    layer_type={'choices': ['Image', 'Labels']},
#    scale={'widget_type' : 'LiteralEvalLineEdit'}, 
#    translate={'widget_type' : 'LiteralEvalLineEdit'}, 
#)

@magic_factory(
    save_dir={'widget_type': 'FileEdit', 'mode' : 'd'}, 
    chunk_size={'widget_type' : 'LiteralEvalLineEdit'}, 
    margin={'widget_type' : 'LiteralEvalLineEdit'},    
)
def assess_segmentation(
    napari_viewer: napari.Viewer,
    ground_truth: napari.layers.Labels, 
    model_segmentation: napari.layers.Labels, 
    chunk_size: tuple = (10, 256, 256), 
    margin: tuple = (1, 64, 64),
    variation_of_information: bool = True, 
    average_precision: bool = True, 
    object_count: bool = True, 
    #diagnostics: bool,
    save_dir: str = 'choose directory', 
    save_prefix: str = 'segmentation-metrics',
    name: str = '', 
    show: bool = True, 
    exclude_chunks_less_than: int = 10,
    ):
    _assess_segmentation(ground_truth, model_segmentation, 
                         chunk_size, margin, variation_of_information, average_precision, 
                         object_count, save_dir, save_prefix, name, show, exclude_chunks_less_than)

    # Diagnostic plots
    # plot_diagnostics(...)
    #TODO:
    # - get image metrics relating to quality/information/randomness/edges
    # - get object metrics for segmentation 


def _assess_segmentation(
    ground_truth, 
    model_segmentation, 
    chunk_size: tuple = (10, 256, 256), 
    margin: tuple = (1, 64, 64),
    variation_of_information: bool = True, 
    average_precision: bool = True, 
    object_count: bool = True, 
    #diagnostics: bool,
    save_dir: str = 'choose directory', 
    save_prefix: str = 'segmentation-metrics',
    name: Union[str, None] = None,
    show: bool = True, 
    exclude_chunks_less_than: int = 10,
    ):
    if name == None:
        name = save_prefix
    # save info
    os.makedirs(save_dir, exist_ok=True)
    data_path = os.path.join(save_dir, save_prefix + '_metrics.csv')
    # need to get the slices from the model-produced layer
    if isinstance(model_segmentation, napari.layers.Labels):
        shape = model_segmentation.data.shape
    else:
        shape = model_segmentation.shape
    slices = get_slices_from_chunks(shape, chunk_size, margin)
    data, stats = model_assessment(
        ground_truth, 
        model_segmentation, 
        save_prefix,
        name,
        slices,
        save_dir,
        variation_of_information, 
        average_precision, 
        object_count, 
        exclude_chunks_less_than)
    # generate plots
    plot_accuracy_metrics(
        data, 
        save_prefix, 
        save_dir, 
        name,
        variation_of_information, 
        average_precision, 
        object_count, 
        show
        )

    # Diagnostic plots
    # plot_diagnostics(...)
    #TODO:
    # - get image metrics relating to quality/information/randomness/edges
    # - get object metrics for segmentation 


def model_assessment(
    ground_truth: napari.layers.Labels, 
    model_segmentation: napari.layers.Labels, 
    save_prefix: str,
    name: str,
    slices: list,
    save_dir: str,
    variation_of_information: bool, 
    average_precision: bool, 
    object_count: bool, 
    exclude_chunks_less_than: int 
    ):
    # save info
    os.makedirs(save_dir, exist_ok=True)
    #data_path = os.path.join(save_dir, save_prefix + f'_{name}_metrics.csv')
    # need to get the slices from the model-produced layer
    data, stats = get_accuracy_metrics(slices, ground_truth, model_segmentation, name, save_prefix,
                                variation_of_information, average_precision, 
                                object_count, save_dir, exclude_chunks_less_than)
    return data, stats


def chunk_n_margin(meta, chunk_size, margin):
    mcs = meta.get('chunk_size')
    if mcs is not None:
        chunk_size = mcs
    mmg = meta.get('margin')
    if mmg is not None:
        margin = mmg
    return chunk_size, margin


def get_slices_from_chunks(arr_shape, chunk_size, margin):
    if len(arr_shape) <= 3:
        ts = range(1)
        fshape = arr_shape
    else:
        ts = range(arr_shape[0])
        fshape = arr_shape[1:]
    chunk_starts, chunk_crops = make_chunks(fshape, chunk_size, margin)
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

# ---------------------------------
# Comparing models or ground truths
# ---------------------------------

@magic_factory(
    comparison_directory={'widget_type': 'FileEdit', 'mode' : 'd'}, 
    fig_size={'widget_type' : 'LiteralEvalLineEdit'}, 
    VI_indexs={'widget_type' : 'LiteralEvalLineEdit'},
    output_directory={'widget_type': 'FileEdit', 'mode' : 'd'}, 
    file_exstention={'choices': ['pdf', 'svg', 'png']},
)
def compare_segmentations(
       comparison_directory: str, 
        save_name: str,
        file_exstention: str ='pdf', 
        output_directory: Union[str, None] = None,
        variation_of_information: bool =True, 
        object_difference: bool =True, 
        average_precision: bool =True, 
        n_rows: int =2, 
        n_col: int =2, 
        comparison_name: str= "Model comparison",
        VI_indexs: tuple =(0, 1), # (0, 0)
        OD_index: int =2, # (0, 1)
        AP_index: int =3, # (1, 0)
        fig_size: tuple =(7, 6), 
        palette: str='Set2',
        top_white_space: float =5, #TODO eventually figure out how to make this a slider 0-100
        left_white_space: float =15, 
        right_white_space: float =5, 
        bottom_white_space: float =10, 
        horizontal_white_space: float =40,
        vertical_white_space: float =40,
        font_size: int =30,
        style: str ='ticks', 
        context: str='paper', 
        show: bool=True
    ):
    '''
    Make a custom plot to compare sementation models based on output from 
    the assess segmentation tool. The only requirement is that files you 
    want to compare are in the same directory, which should be specified
    as the comparison_directory. The plots will be saved into a single 
    file <save_name>.<file_exstention> (e.g., )

    Parameters
    ----------
    comparison_directory: str
        Directory in which to look for the data to plot
    save_name:
        Name to give the output file. Please don't add a file
        extension.
    file_exstention: str (.pdf)
        Specify one of the following file types: .pdf, .png, .svg
    output_directory: str or None (None)
        Directory into which to save the file. If None, 
        will save into the comparison directory. 
    variation_of_information: bool (True)
        Should we plot VI? This will be plotted in 2 plots:
        one for oversegmentation and one for undersegmentation. 
    object_difference: bool (True)
        Should we plot OD? Will be plotted into a single plot. 
    average_precision: bool (True)
        Should we plot AP? Will be plotted into a single plot.
    comparison_name: str ("Model comparison")
        Label to give comparison in plots. Will be used as
        an axis label in OD and VI plots
    n_rows: int (2) 
        How many rows of plots. 1 - 4.
        e.g. - two because we need all four plots for VI, OD, and AP
             - four because we need all four plots for VI, OD, and AP
               and want to plot everthing in the same column. 
    n_col: int =2, 
        How many rows of plots. 1 - 4.
        e.g. - two because we need all four plots for VI, OD, and AP
             - four because we need all four plots for VI, OD, and AP
               and want to plot everthing in the same row. 
        (e.g., two because we need all four plots for VI, OD, and AP)
    VI_indexs: tuple of int or int (0, 1)
        Which plot to put the VI plots in. The first index refers to
        the oversegmentation plot and the second refers to the 
        undersegmentation plot For instructions see "Using integer indexes" 
        in the function Notes below. 
    OD_index: tuple of int or int (2)
        Which plot to put the OD plot in.For instructions see 
        "Using integer indexes". 
    AP_index: tuple of int or int (3)
        Which plot to put the AP plot in.For instructions see 
        "Using integer indexes". 
    fig_size: tuple (9, 9)
        Size of the figure you want to make (in inches). 
    raincloud_orientation: str ("h")
        Orientation for raincloudplots. "h" for horizontal. 
        "v" for vertical. 
    raincloud_sigma: float (0.2)
        The sigma value used to construct the kernel density estimate 
        in the raincloudplot. Determines the smoothness of the 
        half violinplot/density estimate parts of the plot. Larger values
        result in smoother curves. 
    palette: str ('Set2')
        pandas colour palette to use. See the pandas palette documentation
        for details. 
    top_white_space: float (3)
        percent of the total figure size that should remain white space 
        above the plots.
    left_white_space: float (15)
        percent of the total figure size that should remain white space 
        to the left of the plots.
    right_white_space: float (5) 
        percent of the total figure size that should remain white space 
        to the right the plots.
    bottom_white_space: float (17)
        percent of the total figure size that should remain white space 
        below the plots.
    horizontal_white_space: float (16)
        percent of the total figure size that should remain white space 
        between the plots horizontally.
    vertical_white_space: float (16)
        percent of the total figure size that should remain white space 
        between the plots vertically.
    font_size: int (12)
        Size of the axis label and ticks font
    style: str ("ticks")
        Pandas style. Please see pandas documentation for more info and
        options.  
    context: str ("paper")
        Pandas context. Please see pandas documentation for more info and
        options.  
    show: bool (True)
        Do you want to see the plot in the matplotlib viewer once it has 
        been saved?

    Notes
    -----

        Using integer indexes
        ---------------------
        Numbers 0-4 that tells you which position to place the 
        oversegmentation and undersegmentation plots, respectively 
        Note that no matter how rows and columns are arranged, 
        the numbering will start at the top left plot and proceed
        left to write (much like reading English).
        e.g. - for VI plots, (0, 1) in a 2x2 grid of plots will place
               the VI plots in top two plots 
             - for OD plot, 3 in a 1x4 grid will place the OD plot 


    '''
    raincloud_orientation ='h'
    raincloud_sigma =0.2
    comparison_plots(comparison_directory, save_name, file_exstention, 
        output_directory, variation_of_information, 
        object_difference, average_precision, n_rows, n_col, 
        comparison_name, VI_indexs, OD_index, AP_index,
        fig_size, raincloud_orientation, raincloud_sigma,
        palette, top_white_space, left_white_space, 
        right_white_space, bottom_white_space, horizontal_white_space,
        vertical_white_space, font_size, style, context, show)


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
