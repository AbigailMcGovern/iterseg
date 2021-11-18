import ast
from typing import Optional, Dict, Union

import numpy as np
import napari
from napari.qt import thread_worker
from napari_plugin_engine import napari_hook_implementation
from magicgui import widgets, magic_factory
import toolz as tz

from .predict import throttle_function, load_default_unet, predict_output_chunks
from . import watershed as ws


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
                        unet={'widget_type': 'LineEdit', 'value': 'optional/path/to/unet.pt'}, 
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


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    # you can return either a single widget, or a sequence of widgets
    return [UNetPredictWidget, copy_data]
