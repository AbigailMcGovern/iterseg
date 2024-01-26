import ast
from typing import Union
import numpy as np
import napari
from magicgui import magic_factory
from .predict import make_chunks
from .training_experiments import get_experiment_dict, run_experiment
import zarr
from skimage.io import imread
import os
from .metrics import get_accuracy_metrics, plot_accuracy_metrics
from .plots import comparison_plots
from typing import Union
import json
from pathlib import Path
from .segmentation import segmenters
import dask.array as da
from dask import delayed


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
    learning_rate={'widget_type' : 'LiteralEvalLineEdit'}
    )
def train_from_viewer(
    viewer: napari.viewer.Viewer, 
    image_stack: napari.layers.Image, 
    labels_stack: napari.layers.Labels,
    output_dir: Union[str, None]=None, 
    scale: tuple=(1, 1, 1), 
    mask_prediction='mask', 
    centre_prediciton='centreness-log', #lol btw this is a typo in the whole repo :P ... centredness 
    affinities_extent=1, 
    training_name='my-unet',
    loss_function='BCELoss', 
    learning_rate=0.01, 
    epochs=4,
    validation_prop=0.2, 
    n_each=50,
    predict_labels: bool=True,
    save_labels=True,
    ):
    '''
    Train a U-net from the viewer. This will ideally be updated in later releases to allow for other network archetectures/segmentation algorithms to be trained (more in the fashion of segment_data). 

    Parameters
    ----------
    viewer: napari.viewer.Viewer

    image_stack: napari.layers.Image 
    output_dir: str or None (None)
    scale: tuple (1, 1, 1) 
    mask_prediction: str ('mask')
    centre_prediciton: str ('centreness-log')
    affinities_extent: int (1)
    training_name: str ('my-unet')
    loss_function: str ('BCELoss') 
    learning_rate: scalar (0.01)
    epochs: int (4)
    validation_prop: (0.2)
    n_each: (50)
    predict_labels: bool (True)
    save_labels: (True)
    '''
    _train_from_viewer(viewer, image_stack, labels_stack, output_dir, scale, 
        mask_prediction, centre_prediciton, affinities_extent, training_name, 
        loss_function, learning_rate, epochs, validation_prop, n_each, predict_labels, save_labels)


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
    # in case there is an accidental extra dim
    image_4D_stack = np.squeeze(image_4D_stack)
    labels_4D_stack = np.squeeze(labels_4D_stack)
    # make sure the image and gt are the same shape
    assert image_4D_stack.shape == labels_4D_stack.shape
    # if the image is 3D convert to 4D
    if image_4D_stack.ndim == 3:
        image_4D_stack = image_4D_stack[np.newaxis, :, :, :]
        labels_4D_stack = labels_4D_stack[np.newaxis, :, :, :]
    condition_name = [training_name, ]
    image_list = [image_4D_stack[i, ...] for i in range(image_4D_stack.shape[0])]
    labels_list = [labels_4D_stack[i, ...] for i in range(labels_4D_stack.shape[0])]
    del image_4D_stack
    del labels_4D_stack

    # Construct training info
    # -----------------------
    channels_list = construct_channels_list(affinities_extent, mask_prediction, 
                                        centre_prediciton)
    print('Channels list:')
    print(channels_list)
    conditions_list = construct_conditions_list(image_list, loss_function, 
                                                learning_rate, epochs, scale)
    print('Conditions list:')
    print(conditions_list)
    exp_dict = get_experiment_dict(channels_list, condition_name, 
                                   conditions_list=conditions_list, 
                                   validation_prop=validation_prop, 
                                   n_each=n_each)
    print('Experiment dict:')
    print(exp_dict)

    # Run training experiment
    # -----------------------
    u_path = run_experiment(exp_dict, image_list, labels_list, output_dir)

    # Predict full labels
    # -------------------
    if predict_labels:
        if save_labels:
            save_path = os.path.join(str(output_dir), training_name + '_labels-prediction.zarr')
        else:
            save_path = None
        seg_func = segmenters['affinity-unet-watershed']
        seg_func(napari_viewer=viewer, input_volume_layer=image_stack, save_dir=output_dir, 
             name=f'{training_name}_labels', unet_or_config_file=u_path[0], layer_reference=None, 
              chunk_size=(10, 256, 256), margin=(1, 64, 64), debug=False)
        labels_layer = viewer.layers[f'{training_name}_labels']
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
        'output_dir' : str(output_dir), 
        'learning_rate' : learning_rate, 
        'epochs' : epochs, 
        'validation_prop' : validation_prop, 
        'n_each' : n_each, 
        'labels_path' : save_path
    }
    if predict_labels and isinstance(labels_layer, napari.layers.Labels):
        labels_layer.metadata.update(meta)
    json_object = json.dumps(meta, indent=4)
    meta_path = os.path.join(output_dir, Path(u_path[0]).stem + '_meta.json')
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
    layer_type={'choices': ['Image', 'Labels', 'Shapes']},
    scale={'widget_type' : 'LiteralEvalLineEdit'}, 
    translate={'widget_type' : 'LiteralEvalLineEdit'}, 
)
def load_data(
    napari_viewer: napari.viewer.Viewer, 
    layer_name:str,
    layer_type: str='Image',
    data_type: str ='individual frames',
    directory: Union[str, None] =None, 
    data_file: Union[str, None] =None,
    scale: tuple =(1, 1, 1),
    translate: tuple =(0, 0, 0),
    split_channels: bool=False, 
    in_memory: bool=True,
    ):
    '''
    Load the data into the viewer as a stack of 3D image frames. 

    Parameters
    ----------
    layer name: str (None)
        What will you call the layer of images you want to load in?
    layer_type: str ('Image')
        Is the data you want to load a segmentation (use "Labels") or an image (use "Image")
    data_type: str ('individual frames')
        Are the data a series of individual 3D images? If so use "individual images". If the 
        data are in the format of a single 4D or 5D image also use "individual images". If 
        you have a directory with several 4D images, use "image stacks". 
    directory:
        Path to the a directory containing image stacks (4D) or individual frames (3D) or
        to a zarr file. Data can be 3D (ZYX), 4D (TZYX), or 5D (CTZYX). Zarr files must
        end in .zarr/.zar. If you want to load a single tiff file, leave blank. 
    data_file: str (None)
        Path to the image stack (if data_type is "image stacks") or to 
        a single image frame (if data_type is "individual frames"). 
        This needs to end in .tiff/.tif. If you want to load from a directory or 
        want to load a zarr, leave blank. 
    scale: tuple of float
        Scale of the image/segmentation to display in z, y, x format.
    translate: tuple of float
        Translation of the image origin in z, y, x format.
    split_channels:
        If you have a multichannel image in the format CTZYX (or any format really as
        long as the first dim is the channel) you can split these. This is useful if 
        you are loading 5D data to segment. 
    in_memory:
        Do you want all of the data to be loaded into your computer's memory (i.e., 
        in RAM). Select false if your data is bigger than RAM or very large.
    save_stack:
        If you loaded a stack of images, do you want to save these as a single stack 
        of zarr files. If you do this for a stack of labels, you will be able to paint
        directly into the files when correcting labels. 
    '''
    _load_data(napari_viewer, layer_name, 
               layer_type, data_type,
                 directory, data_file,
                scale, translate, split_channels, 
                in_memory)


def _load_data(
    napari_viewer: napari.viewer.Viewer, 
    layer_name: str,
    layer_type: str,
    data_type: str ='individual frames',
    directory: Union[str, None] =None, 
    data_file: Union[str, None] =None,
    scale: tuple =(1, 1, 1),
    translate: tuple =(0, 0, 0),
    split_channels: bool=False, 
    in_memory: bool=True,
    ):
    '''
    Load the data into the viewer as a stack of 3D image frames. 

    Parameters
    ----------
    layer name: str (None)
        What will you call the layer of images you want to load in?
    layer_type: str ('Image')
        Is the data you want to load a segmentation (use "Labels") or an image (use "Image")
    data_type: str ('individual frames')
        Are the data a series of individual 3D images? If so use "individual images". If the 
        data are in the format of a single 4D or 5D image also use "individual images". If 
        you have a directory with several 4D images, use "image stacks". 
    directory:
        Path to the a directory containing image stacks (4D) or individual frames (3D) or
        to a zarr file. Data can be 3D (ZYX), 4D (TZYX), or 5D (CTZYX). Zarr files must
        end in .zarr/.zar. If you want to load a single tiff file, leave blank. 
    data_file: str (None)
        Path to the image stack (if data_type is "image stacks") or to 
        a single image frame (if data_type is "individual frames"). 
        This needs to end in .tiff/.tif. If you want to load from a directory or 
        want to load a zarr, leave blank. 
    scale: tuple of float
        Scale of the image/segmentation to display in z, y, x format.
    translate: tuple of float
        Translation of the image origin in z, y, x format.
    split_channels:
        If you have a multichannel image in the format CTZYX (or any format really as
        long as the first dim is the channel) you can split these. This is useful if 
        you are loading 5D data to segment. 
    in_memory:
        Do you want all of the data to be loaded into your computer's memory (i.e., 
        in RAM). Select false if your data is bigger than RAM or very large.
    save_stack:
        If you loaded a stack of images, do you want to save these as a single stack 
        of zarr files. If you do this for a stack of labels, you will be able to paint
        directly into the files when correcting labels. 
    '''
    if directory is not None:
        directory = str(directory)
    if data_file is not None:
        data_file = str(data_file)
    if layer_type == 'Image' or layer_type == 'Labels':
        imgs, uses_directory = read_data(directory, data_file, data_type, in_memory)
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
    if layer_type == 'Shapes':
        if data_file is not None:
            shapes = read_shapes(data_file)
        elif directory is not None:
            files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npy')]
            shapes = []
            for f in files:
                new = read_shapes(f)
                shapes = shapes + new
        else:
            raise ValueError("Please ensure you pick a file or directory to read from")
        napari_viewer.add_shapes(shapes, scale=scale, name=layer_name, translate=translate)


def read_shapes(data_file):
    shapes = np.load(str(data_file), allow_pickle=False)
    shapes = [s for s in shapes]
    return shapes

# TODO: refactor this **awful** function... you can do better
def read_data(directory, data_file, data_type, in_memory):
    """
    When supplied with a directory that is a zarr, this function will open
    the image as a zarr (not in memory). When supplied with a directory
    of files, the function will open any tiffs or zarrs in the directoy
    as a 4D stack of 3D images. When supplied with a single file that is
    a tiff, the tiff will be loaded (regardless of dimension). 

    If in_memory, the data will be opened as a numpy array. If this is 
    set to False, the data will be opened as a dask array. This is neccessary
    if your image is too big to fit in the computer's RAM. 
    """
    possible_suf = ['.zarr', '.zar', '.tiff', '.tif']
    # is the data coming from a directory (not a zarr file)
    uses_directory = directory is not None
    is_zarr = False
    if uses_directory:
        uses_directory = os.path.isdir(directory) and not directory.endswith('.zarr') and not directory.endswith('.zar')
    # is the data coming from a single file
    single_file = data_file is not None
    if single_file:
        if data_file.endswith('.tiff') or data_file.endswith('.tif'):
            data_paths = [data_file, ]
    elif not uses_directory:
        is_zarr = directory.endswith('.zarr') or directory.endswith('.zar')
    elif uses_directory:
        data_paths = []
        for f in os.listdir(directory):
            bool_list = [f.endswith(s) for s in possible_suf]
            if True in bool_list:
                data_paths.append(os.path.join(directory, f))
    # Read a zarr
    # -----------
    if is_zarr:
        if directory.endswith('.ome.zarr'):
            p = os.path.join(directory, '0')
            imgs = zarr.open(p)
        else:
            imgs = zarr.open(directory)
    # Everything that isnt a zarr 
    # ---------------------------
    else:
        imgs = []
        data_paths = sorted(data_paths)
        if in_memory:
            lazy_imread = None
        else:
            lazy_imread = delayed(imread)
        for p in data_paths:
            im = read_with_correct_modality(p, in_memory, lazy_imread)
            imgs.append(im)
        imgs = correct_shape(imgs)
        # Read in from a directory of files
        # ---------------------------------
        if uses_directory:
            if not in_memory:
                sample = imgs[0].compute
                [da.from_delayed(delayed_reader, shape=sample.shape, dtype=sample.dtype)
                for delayed_reader in imgs]
                del sample
            # if data is in stack/s (4D stacks of 3D frames)
            if data_type == 'image stacks' and len(imgs) > 1:
                if in_memory:
                    imgs = np.concatenate(imgs)
                else:
                    imgs = da.concatenate(imgs)
            # if data is in individual 3D frames
            if data_type == 'individual frames':
                if in_memory:
                    imgs = np.stack(imgs)
                else:
                    imgs = da.stack(imgs)
        # Single image - not zarr
        # -----------------------
        else:
            if in_memory:
                imgs = imgs[0]
            else:
                imgs = imgs[0].compute
                if imgs.ndim > 3:
                    chunks = (1, ) * imgs.ndim - 3 + imgs.shape[-3:]
                else:
                    chunks = 'auto'
                imgs = da.from_array(imgs, chunks=chunks)
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


def read_with_correct_modality(path, in_memory, lazy_imread):
    if in_memory:
        if path.endswith('.tif') or path.endswith('.tiff'):
            im = imread(path)
        elif path.endswith('.ome.zarr'):
            p = os.path.join(path, '0')
            im = zarr.creation.open_array(p, 'r')
            im = np.array(im)
        elif path.endswith('.zar') or path.endswith('.zarr'):
            im = zarr.creation.open_array(path, 'r')
            im = np.array(im)
    else:
        if path.endswith('.tif') or path.endswith('.tiff'):
            im = lazy_imread(path)
    return im



def correct_shape(imgs):
    imgs = [np.squeeze(img) for img in imgs]
    shapes_3D = np.array([im.shape[-3:] for im in imgs])
    shape_3D = np.max(shapes_3D, axis=0)
    #max_size = [s == shape_3D for s in shapes_3D]
    not_max_size = [s != shape_3D for s in shapes_3D]
    if np.sum(not_max_size) > 0:
        shapes = np.array([im.shape for im in imgs])
        final_shapes = [list(s[:-3]) + list(shape_3D) for s in shapes]
        dtype = imgs[0].dtype
        final_imgs = [np.zeros(s, dtype=dtype) for s in final_shapes]
        slices = [tuple([slice(0, s) for s in shape]) for shape, im in zip(shapes, imgs)]
        for s_, img, orig in zip(slices, final_imgs, imgs):
            img[s_] = orig
        return final_imgs
    else:
        return imgs



# --------------
# Segment widget
# --------------

@magic_factory(
        save_dir={'widget_type': 'FileEdit', 'mode' : 'd'}, 
        chunk_size={'widget_type': 'LiteralEvalLineEdit'}, 
        margin={'widget_type': 'LiteralEvalLineEdit'}, 
        segmenter={'choices': list(segmenters.keys())}, 
        network_or_config_file={'widget_type': 'FileEdit'}, 
)
def segment_data(
    napari_viewer: napari.Viewer, 
    input_volume_layer: napari.layers.Image,
    save_dir: Union[str, None] = None,
    name: str = 'labels-prediction',
    segmenter: str='affinity-unet-watershed', 
    network_or_config_file: Union[str, None] = None,
    layer_reference: Union[str, None] = None,
    chunk_size: tuple = (10, 256, 256),
    margin: tuple = (1, 64, 64),
    debug: bool = True # TODO change to false
    ):
    '''
    Parameters
    ----------
    napari_viewer: napari.Viewer
        The napari viewer - this is automatically passed to magicgui.
    input_volume_layer: napari.layers.Image
        The image layer that you want to segment. 
    save_dir: str or None (None)
        The directory into which you want to save the data. If you
        do not choose one it will not be saved. Omit at own risk... 
    name: str ('labels-prediction')
        The name of the output layer and output file. 
    segmenter: str ('affinity-unet-watershed')
        This is the segmentation algorith you want to use. We have several 
        options:
            - affinity-unet-watershed: corresponds to the feature-based 
              segmentation trained using train_from_viewer. The unet predicts
              the object boundaries in each of three axes (affinities), 
              a mask (foreground), and the location of object centres. 
              These feature maps are used to segment the image with a
              modified watershed algorith.
            - unet-mask: segment with a unet to get only a single output
              channel with a mask. Will be added as an option for training
              in a future version. At the present this can be used to obtain
              only the mask from the affinity-unet-watershed. 
            - otsu-mask: a mask derived from smoothing the image with a 
              gaussian kernel then using otsu thresholding. To be added soon. 
            - blob-watershed: classical blob detection using a Laplacian of 
              Gaussian kernel to find object coordinate and inverse image intensity
              to segment the image using a cannonical watershed.
    network_or_config_file: str or None (None)
        A path to a neural network to use for the segmentation. If None and 
        using affinity-unet-watershed, the defaul U-net will be used. It is 
        trained to segment platelets. 
    layer_reference: str or None (None)
        A layer that the segmentation algorithm can get information from. 
        For instance, if the train_from_viewer function is used, it will 
        add a unet path to the layer metadata for any predicted labels.
        This can be accessed by affinity-unet-watershed. 
        TODO make sure this works!
    cchunk_size: tuple = (10, 256, 256) 
        What size of chunks would you like to compute scores for?  
        Each chunk is represented as a point on the VI and object count plots.
    margin: tuple = (1, 64, 64)
        This is the margin or overlap between chunks. The default is set this 
        way because this replicates the chunks that images might were segmented in. 
        The margin will change the number and position of chunks created. 
    debug: bool (False)
        Do you need to debug the code or save intermediate Unet output to check?
        If you use this, you will not be able to use the viewer until the code is 
        finished running or until it crashes. The reason for this is that the code 
        is run in the same thread as the viewer so  errors will be caught propery.
    '''
    seg_func = segmenters[segmenter]
    seg_func(napari_viewer, input_volume_layer, save_dir, 
             name, network_or_config_file, layer_reference, 
             chunk_size, margin, debug)


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
    save_dir: Union[str, None] = None, 
    save_prefix: str = 'segmentation-metrics',
    name: Union[str, None] = None, 
    show: bool = True, 
    exclude_chunks_less_than: int = 10,
    ):
    '''
    napari_viewer: napari.Viewer
        The napari viewer - this is automatically passed to magicgui.
    ground_truth: napari.layers.Labels
        The ground truth that you want to use to assess the segmentation.
    model_segmentation: napari.layers.Labels
        The segmentation you want to assess. 
    chunk_size: tuple = (10, 256, 256) 
        What size of chunks would you like to compute scores for?  
        Each chunk is represented as a point on the VI and object count plots.
    margin: tuple = (1, 64, 64)
        This is the margin or overlap between chunks. The default is set this 
        way because this replicates the chunks that images might were segmented in. 
        The margin will change the number and position of chunks created. 
    variation_of_information: (True)
        Will you assess VI?
        The variation of information is defined as VI(X,Y) = H(X|Y) + H(Y|X). 
        If X is the ground-truth segmentation, then H(X|Y) can be interpreted as 
        the amount of under-segmentation and H(Y|X) as the amount of 
        over-segmentation. In other words, a perfect over-segmentation will have 
        H(X|Y)=0 and a perfect under-segmentation will have H(Y|X)=0.
    average_precision: bool (True)
        Will you assess AP?
        Average precision is a combined measure of how accurate the model is at finding 
        true positive (real) objects (we call this precision) and how many of ground 
        truth real objects it found (this is called recall). 
            Precision = TP / (TP + FP)
            Recall = TP / (TP + FN)
            Abbreviations: FN, false negative; TP, true positive; FP, false positive 
        The assessment of whether an object is TP, FP, and FN depends on the threashold
        of overlap between objects. Here we use the intersection of union (IoU), which is
        the proportion of overlap between the bounding boxes of ground truth and model
        segemented objects. AP is assessed using different IoU thresholds (from 0.35-0.95). 
        The resultant data will be plotted as IoU by AP. 
    object_count: bool (True)
        Will you assess object count difference?
        The object count difference is simply the number of objects difference between the
        ground truth and the segementation. 
            card(ground truth) - card(segmentation)
        With this, the number of objects in ground truth and segementation + the percentage
        difference are added to the final data frame. 
    save_dir: str (None)
        This is the directory to which you will save the output. The assessment will not 
        run if you don't choose one. Ask yourself, why run an analysis without saving 
        anything? 
    save_prefix: str ('segmentation-metrics')
        This is the prefix that will be used to save the data. Several files will be saved:
            - <save_prefix>_AP_curve.csv (AP data)
            - <save_prefix>_scores.csv (VI, OC, and data used to create AP)
            - <save_prefix>_stats.csv (descriptive stats and 95% CI for variable is scores file)
    name: str (None)
        This is the name to give the model. This will be added in the data sheet in the column
        model_name. This is important because it is used in compare_segmentations to assign the 
        data to a group that can be compared to other groups. If None, the save_prefix will be
        used.   
    show: bool (True)
        Do you want to see the plots at the end?
    exclude_chunks_less_than: int (10)
        What is the minimum number of objects you need to see in a chunk of ground truth data
        to assess the chunk? Only use this if you KNOW you don't care about regions of the image 
        with few objects. You will only know this by visually inspecting the data + understanding
        your use case. 
    '''
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
    assert save_dir is not None, 'Please pick a directory to which to save the data.'
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


# -----------
# Save frames
# -----------

@magic_factory(
    save_dir={'widget_type': 'FileEdit', 'mode' : 'd'}, 
    frames={'widget_type' : 'LiteralEvalLineEdit'},
)
def save_frames(
        napari_viewer: napari.Viewer, 
        layer: napari.layers.Layer,
        save_dir: Union[str, None]=None,
        save_name: Union[str, None]=None,
        frames: Union[tuple, int]=None,
        save_as_stack: bool=True,
        load_saved: bool=False, 
        load_name: Union[str, None]=None,
    ):
    '''
    Save data from frames

    Parameters
    ----------
    napari_viewer: napari.Viewer
    layer: napari.layers.Layer
        The layer to save data from. Supported layers include:
        Image, Labels, Shapes, and Points. 
    save_dir: str
        The directory into which to save the data
    save_name:
        The name you want to use to save the data. This doesn't 
        need an extension. If a labels or image layer, data will
        be saved as a zarr file. If a shapes or points layer, 
        data will be saved as a npy file. 
    frames: tuple or None
        The frames you want to save (in brackets like so: (0, 2, 3))
        or None. If None, the whole layer will be saved. This will 
        only work for Labels and Image layers. All frames will always
        be saved for Points and Shapes layers. 
    
    '''
    #if isinstance() # check if dask array
    if isinstance(frames, int):
        frames = (frames, )
    if isinstance(layer, napari.layers.Image) or isinstance(layer, napari.layers.Labels):
        if isinstance(frames, tuple):
            slices = [slice(f, f+1) for f in frames]
            data = [layer.data[s] for s in slices]
            if save_as_stack:
                data = np.stack(data)
                data = np.squeeze(data)
                sp = os.path.join(str(save_dir), save_name + '.zarr')
                zarr.save(sp, data)
            else:
                for f, d in zip(frames, data):
                    sn = f'{save_name}_f{f}'
                    sp = os.path.join(str(save_dir), sn + '.zarr')
                    zarr.save(sp, d)
        if frames is None:
            if save_as_stack:
                sp = os.path.join(str(save_dir), save_name + '.zarr')
                data = np.squeeze(layer.data)
                zarr.save(sp, data)
            else:
                for f in layer.data.shape[0]:
                    sn = f'{save_name}_f{f}'
                    sp = os.path.join(str(save_dir), sn + '.zarr')
                    zarr.save(sp, layer.data[f, ...])
    elif isinstance(layer, napari.layers.Shapes):
        #TODO add shapes saving capabilities for ROIs
        data = layer.data
        data = np.stack(data)
        sp = os.path.join(str(save_dir), save_name + '.npy')
        np.save(sp, data, allow_pickle=False)
    elif isinstance(layer, napari.layers.Points):
        data = layer.data
        sp = os.path.join(str(save_dir), save_name + '.npy')
        np.save(sp, data, allow_pickle=False)
    load_saved_data(load_saved, napari_viewer, frames, layer, sp, load_name)
            


def load_saved_data(load_saved, napari_viewer, frames, layer, sp, load_name):
    if load_saved:
        if isinstance(layer, napari.layers.Image) or isinstance(layer, napari.layers.Labels):
            loaded = zarr.open(sp)
        elif isinstance(layer, napari.layers.Shapes) or isinstance(layer, napari.layers.Points):
            loaded = np.load(sp, allow_pickle=False)
        if load_name is None:
            fstr = '-'.join([str(f) for f in frames])
            load_name = f'{layer.name}_f{fstr}'
        if layer.ndim != loaded.ndim:
            diff = layer.ndim - loaded.ndim
            take_from = diff - layer.ndim
            scale = layer.scale[take_from:]
        else:
            scale = layer.scale
        if isinstance(layer, napari.layers.Image):
            napari_viewer.add_image(loaded, name=load_name, scale=scale)
        elif isinstance(layer, napari.layers.Labels):
            napari_viewer.add_labels(loaded, name=load_name, scale=scale)
        elif isinstance(layer, napari.layers.Shapes):
            napari_viewer.add_shapes(loaded, name=load_name, scale=scale)
        elif isinstance(layer, napari.layers.Points):
            napari_viewer.add_points(loaded, name=load_name, scale=scale)



# --------------------
# Generate GT from ROI
# --------------------

@magic_factory(
    save_dir={'widget_type': 'FileEdit', 'mode' : 'd'}, 
)
def ground_truth_from_ROI(
        napari_viewer: napari.Viewer, 
        image_layer: napari.layers.Image,
        labels_layer: napari.layers.Labels,
        shapes_layer: napari.layers.Shapes, 
        save_dir: Union[str, None]=None, 
        name: str='gt-from-ROI',
        number_of_tiles: int=1,
        padding: int=2
    ):
    '''
    Generate a ground truth image for training using
    several proofread ROIs.

    Parameters
    ----------
    napari_viewer: napari.Viewer
        The napari viewer. This is passed in automatically. 
    labels_layer: napari.layers.Labels
        The labels layer you want tp create ground truth from.
    shapes_layer: napari.layers.Shapes
        The shapes layer you want to use to find the ROIs. 
        At present, the shapes are treated as rectangles 
        across the xy plane. We will use the entire z axis.
    save_dir: str or None
        Folder to save data to. If this is not None, the data will
        be saved as a zarr file and the data will be opened into a 
        new layer. This means you can continue to proofread and this
        will be saved. 
    name: str='gt-from-ROI'
        Name that will be used for the output layer and for the save
        file. Note that this does not require a file extension. 
    number_of_tiles: int (1)
        The number of times each ROI will be repeated. Each will be saved 
        into its own frame and tiled as many times as specified. Note that
        if the number specifed is greater than the maximum times it fits
        into the frame, the number of tiles will be set to the maximum.
    padding: int (2)
        The number of pixels padding between each tile in the x and y axes. 
    '''
    _ground_truth_from_ROI(napari_viewer, image_layer, labels_layer, shapes_layer, 
                           save_dir, name, number_of_tiles, padding)


def _ground_truth_from_ROI(
        napari_viewer: napari.Viewer, 
        image_layer: napari.layers.Image, # TODO:save the image also with background noise
        labels_layer: napari.layers.Labels,
        shapes_layer: napari.layers.Shapes, 
        save_dir: Union[str, None]=None, 
        name: str='gt-from-ROI',
        number_of_tiles: int=1,
        padding: int=2
    ):
    '''
    Generate a ground truth image for training using
    several proofread ROIs.

    Parameters
    ----------
    napari_viewer: napari.Viewer
        The napari viewer. This is passed in automatically. 
    labels_layer: napari.layers.Labels
        The labels layer you want tp create ground truth from.
    shapes_layer: napari.layers.Shapes
        The shapes layer you want to use to find the ROIs. 
        At present, the shapes are treated as rectangles 
        across the xy plane. We will use the entire z axis.
    save_dir: str or None
        Folder to save data to. If this is not None, the data will
        be saved as a zarr file and the data will be opened into a 
        new layer. This means you can continue to proofread and this
        will be saved. 
    name: str='gt-from-ROI'
        Name that will be used for the output layer and for the save
        file. Note that this does not require a file extension. 
    number_of_tiles: int (1)
        The number of times each ROI will be repeated. Each will be saved 
        into its own frame and tiled as many times as specified. Note that
        if the number specifed is greater than the maximum times it fits
        into the frame, the number of tiles will be set to the maximum.
    padding: int (2)
        The number of pixels padding between each tile in the x and y axes. 
    '''
    gt = labels_layer.data
    img = image_layer.data
    rois = shapes_layer.data
    max_lim = [np.round(np.max(roi, axis=0)).astype(int) + 1 for roi in rois]
    min_lim = [np.round(np.min(roi, axis=0)).astype(int) for roi in rois]
    # Get the ROI labels
    # ------------------
    extra_dims = gt.ndim - 3
    slices = []
    for ll, ul in zip(min_lim, max_lim):
        s_xy = [slice(ll[i], ul[i]) for i in range(extra_dims + 1, gt.ndim)]
        s_z = [slice(None), ]
        s_extra = [slice(ll[i], ul[i]) for i in range(0, extra_dims)] * extra_dims 
        s = s_extra + s_z + s_xy
        s = tuple(s)
        slices.append(s)
    gt_rois = [gt[s_] for s_ in slices]
    im_rois = [img[s_] for s_ in slices]
    # Tile the ROIs as specified
    # --------------------------
    gt_xy_shape = gt.shape[-2:]
    max_per_axis = []
    max_per_axis = [np.floor_divide(gt_xy_shape, 
                        np.array(d.shape[-2:]) + padding).astype(int) \
                            for d in gt_rois]
    max_total = [mpa[0] * mpa[1] for mpa in max_per_axis]
    mod_per_axis = [np.mod(gt_xy_shape, 
                        (np.array(d.shape[-2:]) + padding)).astype(int) \
                            for d in gt_rois]
    final_gt_data = []
    final_im_data = []
    gt_3D_shape = gt.shape[-3:]
    for i, gt_roi in enumerate(gt_rois):
        mt = max_total[i]
        if number_of_tiles < mt:
            mt = number_of_tiles # we wont go past the max possible
        mapa = max_per_axis[i]
        y1 = gt_roi.shape[-2]
        x1 = gt_roi.shape[-1]
        slices = []
        for j in range(0, mt):
            # TODO add image and random background noise
            x_multi_factor = j % mapa[-1] + 1
            y_multi_factor = np.floor_divide(j, mapa[-1]) + 1
            px = (padding * x_multi_factor) - padding
            py = (padding * y_multi_factor) - padding
            _y1 = y1 * y_multi_factor + py
            _x1 = x1 * x_multi_factor + px
            y0 = _y1 - y1
            x0 = _x1 - x1
            s = (slice(None), slice(y0, _y1), slice(x0, _x1))
            slices.append(s)
        new_gt_frame = np.zeros(gt_3D_shape, dtype=gt.dtype)
        new_im_frame = np.random.normal(img.mean(), size=gt_3D_shape)
        #np.zeros(gt_3D_shape, dtype=img.dtype)
        for s in slices:
            new_gt_frame[s] = gt_roi
            new_im_frame[s] = im_rois[i]
        final_gt_data.append(new_gt_frame)
        final_im_data.append(new_im_frame)
    final_gt_data = np.stack(final_gt_data)
    final_gt_data = np.squeeze(final_gt_data)
    final_im_data = np.stack(final_im_data)
    final_im_data = np.squeeze(final_im_data)
    # Save if appropriate
    # -------------------
    if save_dir is not None:
        sp_l = os.path.join(save_dir, name + '_labels.zarr')
        zarr.save(sp_l, final_gt_data)
        sp_i = os.path.join(save_dir, name + '_img.zarr')
        zarr.save(sp_i, final_im_data)
        final_gt_data = zarr.open(sp_l)
        final_im_data = zarr.open(sp_i)
    # Add to viewer
    # -------------
    napari_viewer.add_image(final_im_data, scale=labels_layer.scale, 
                         translate=labels_layer.translate, name=name + '_img')
    napari_viewer.add_labels(final_gt_data, scale=labels_layer.scale, 
                         translate=labels_layer.translate, name=name + '_labels')
  

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



