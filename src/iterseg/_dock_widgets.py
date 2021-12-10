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

@magic_factory(
    image_path={'widget_type': 'FileEdit'},
    labels_dir={'widget_type': 'FileEdit'},
)
def initiate_training_data(image_path, labels_dir):
    # load a single image into 4D array and initiate a labels layer (in file as tesnorstore)
    pass 


@magic_factory(
    images_path={'widget_type': 'FileEdit'}, 
    max_number={'widget_type' : 'LiteralEvalLineEdit'}
)
def add_image_data(images_path, max_number=None):
    pass



# -------------------
# Segement with Model
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


def predict_output_chunks_widget(
        napari_viewer,
        input_volume_layer: napari.layers.Image,
        chunk_size: str = '(10, 256, 256)',
        margin: str = '(0, 0, 0)',
        unet: str = 'default', 
        use_default_unet: bool = True,
        auto_call_watershed: bool = True,
        state: Dict = None,
        ):
    if type(chunk_size) is str:
        chunk_size = ast.literal_eval(chunk_size)
    if type(margin) is str:
        margin = ast.literal_eval(margin)
    if state is None:
        state = {}
    viewer = napari_viewer
    layer = input_volume_layer
    ndim = len(chunk_size)
    slicing = viewer.dims.current_step[:-ndim]
    state['slicing'] = slicing
    input_volume = np.asarray(layer.data[slicing]).astype(np.float32)
    input_volume /= np.max(input_volume)
    if 'unet-output' in state:  # not our first rodeo
        state['unet-worker'].quit()  # in case we are running on another slice
        if state['self'].call_watershed is not None:
            state['self'].call_watershed.enabled = False
        output_volume = state['unet-output']
        output_volume[:] = 0
        layerlist = state['unet-output-layers']
        for layer in layerlist:
            layer.refresh()
    else:
        output_volume = np.zeros((5,) + input_volume.shape, dtype=np.float32)
        state['unet-output'] = output_volume
        scale = np.asarray(layer.scale)[-ndim:]
        translate = np.asarray(layer.translate[-ndim:])
        offsets = -0.5 * scale * np.eye(5, 3)  # offset affinities, not masks
        layerlist = viewer.add_image(
                output_volume,
                channel_axis=0,
                name=['z-aff', 'y-aff', 'x-aff', 'mask', 'centroids'],
                scale=scale,
                translate=list(translate + offsets),
                colormap=[
                        'bop purple',
                        'bop orange',
                        'bop orange',
                        'bop blue',
                        'gray',
                        ],
                visible=[False] * 4 + [True],
                )
        state['unet-output-layers'] = layerlist
        state['scale'] = scale
        state['translate'] = translate
    return_callbacks = [state['self'].add_watershed_widgets]
    if auto_call_watershed:
        return_callbacks.append(lambda _: state['self'].call_watershed())

    def clear_volume(event=None):
        output_volume[:] = 0
        for ly in layerlist:
            ly.refresh()
    # 
    launch_prediction_worker = thread_worker(
            predict_output_chunks,
            connect={
                    'yielded': [ly.refresh for ly in layerlist],
                    'returned': return_callbacks,
                    'aborted': clear_volume,
                    }
            )
    worker = launch_prediction_worker(
            unet, input_volume, chunk_size, output_volume, margin=margin, use_default_unet=use_default_unet
            )
    state['unet-worker'] = worker
    current_step = viewer.dims.current_step
    currstep_event = viewer.dims.events.current_step

    @self_destructing_callback(disconnect=currstep_event.disconnect)
    def quit_worker(event):
        new_step = event.value
        if new_step[:-ndim] != current_step[:-ndim]:  # new slice
            worker.quit()

    currstep_event.connect(quit_worker)
    clear_once = self_destructing_callback(
            clear_volume, currstep_event.disconnect
            )
    currstep_event.connect(clear_once)


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


def segment_from_prediction_widget(
        napari_viewer: napari.viewer.Viewer,
        prediction: np.ndarray,
        state: Optional[Dict] = None,
        ):
    viewer = napari_viewer
    output = np.pad(
            np.zeros(prediction.shape[1:], dtype=np.uint32),
            1,
            mode='constant',
            constant_values=0,
            )
    ndim = output.ndim
    crop = tuple([slice(1, -1),] * ndim)  # yapf: disable
    output_layer = state.get('output-layer')
    if output_layer is None or output_layer not in napari_viewer.layers:
        output_layer = viewer.add_labels(
                output[crop],
                name='watershed',
                scale=state['scale'],
                translate=state['translate'],
                )
        state['output-layer'] = output_layer
    else:
        output_layer.data = output[crop]

    def clear_output(event=None):
        output[:] = 0
        output_layer.refresh()

    launch_segmentation = thread_worker(
            ws.segment_output_image,
            connect={'finished': output_layer.refresh},
            )
    worker = launch_segmentation(
            prediction,
            affinities_channels=(0, 1, 2),
            thresholding_channel=3,
            centroids_channel=4,
            out=output.ravel(),
            )
    current_step = viewer.dims.current_step
    currstep_event = viewer.dims.events.current_step

    @self_destructing_callback(disconnect=currstep_event.disconnect)
    def quit_worker_and_clear(event):
        new_step = event.value
        if new_step[:-ndim] != current_step[:-ndim]:  # new slice
            worker.quit()

    currstep_event.connect(quit_worker_and_clear)
    clear_once = self_destructing_callback(
            clear_output, currstep_event.disconnect
            )
    currstep_event.connect(clear_once)


class UNetPredictWidget(widgets.Container):
    def __init__(self, napari_viewer):
        self._state = {'self': self}
        super().__init__(labels=False)
        self.predict_widget = widgets.FunctionGui(
                predict_output_chunks_widget,
                param_options=dict(
                        napari_viewer={'visible': False},
                        chunk_size={'widget_type': 'LiteralEvalLineEdit'},
                        state={'visible': False},
                        unet={'widget_type': 'FileEdit'}, 
                        use_default_unet={'widget_type': 'CheckBox'} 
                        )
                )
        self.append(widgets.Label(value='U-net prediction'))
        self.append(self.predict_widget)
        self.predict_widget.state.bind(self._state)
        self.predict_widget.napari_viewer.bind(napari_viewer)
        self.viewer = napari_viewer
        self.call_watershed = None

    def add_watershed_widgets(self, volume):
        if self.call_watershed is None:
            self.call_watershed = widgets.FunctionGui(
                    segment_from_prediction_widget,
                    call_button='Run Watershed',
                    param_options=dict(
                            prediction={'visible': False},
                            state={'visible': False},
                            )
                    )
            self.append(widgets.Label(value='Affinity watershed'))
            self.append(self.call_watershed)
        self.call_watershed.prediction.bind(volume)
        self.call_watershed.state.bind(self._state)
        self.call_watershed.enabled = True



# -------------------
# Validation Analysis
# -------------------

@magic_factory()
def validation_analysis(
    images: napari.types.ImageData,
    model_1: napari.types.LabelsData,
    model_2: napari.types.LabelsData,
    model_3: napari.types.LabelsData,
    model_4: napari.types.labelsData, 
    ):
    # want to compute quality metrics etc for every chunk of data
    # passed through the pipeline 
    # therefore need the unet chunks info
    # probs get input chunk info from unet state dict
    pass



# -------------------
# Hook implementation
# -------------------


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    # you can return either a single widget, or a sequence of widgets
    return [UNetPredictWidget, copy_data, train_from_viewer, load_train_data]
