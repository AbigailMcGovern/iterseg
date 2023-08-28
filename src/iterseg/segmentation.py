import numpy as np
from skimage.segmentation import watershed
from skimage.feature import peak_local_max, blob_log, blob_dog
from skimage.filters import gaussian
from typing import Union, Callable
import napari
import json
from iterseg.predict import process_chunks, predict_chunk_feature_map, load_unet
from iterseg import watershed as ws
import zarr
import os
from napari.qt import thread_worker
from scipy import ndimage as ndi
import pathlib
import os


# ------------------------
# Affinity U-net Watershed
# ------------------------

def affinity_unet_watershed(
        napari_viewer, 
        input_volume_layer: napari.layers.Image,
        save_dir: Union[str, None] = None,
        name: str = 'my-segmentation',
        unet_or_config_file: Union[str, None] = None,
        layer_reference: Union[str, None] = None,
        chunk_size: Union[tuple, None] = (10, 256, 256),
        margin: Union[tuple, None] = (1, 64, 64),
        debug: bool =False
    ):
    '''
    Segment a 3D image or stack of 3D images using a affinity-unet-watershed
    algorithm. The image is passed to a U-Net, which outputs 5 feature maps:
    (1) z-axis affinities, (2) y-axis affinities, (3) x-axis affinities, 
    (4) foreground mask, (5) centre point predicition. The feature maps are
    used in a modified watershed algorithms. The centre point prediciton is
    used to find object centres and the foreground mask is used to determine
    the pixels that need to be labeled. The affinities in each axis demarkate
    the boundaries between objects and the waterhshed uses these to find the 
    edges of objects. 

    Parameters
    ----------
    napari_viewer: napari.Viewer
        The viewer into which the resultant segmentation will be placed.
    input_volume_layer: napari.layers.Image
        The viewer layer with the image to be processed.
    save_dir: str or None (None)
        The directory into which to save the outputted segmentation. 
    name: str ('labels-prediction')
        The name with which to save the data. Also the name of the 
        resultant segmentation layer that will be displayed in the 
        viewer. 
    config_file: str or None (None)
        Path to a json file with any additional parameters. I'm sure
        There is  a more elegant way to do this through the GUI, 
        but I don't have the time to work on this for now. 
    layer_reference: str or None (None)
        If you need information from one of the layers, this is the
        layer to get it from. This is fed into the custom config function. 
        This function can get a unet from layer metadata. 
    chunk_size: tuple (10, 256, 256)
         This tells you how big the chunks to be processed are
    margin: tuple (1, 64, 64)
        This tells you how much overlap between chunks is used
    '''
    segmentation_wrapper(affinity_watershed_for_chunks, affinity_watershed_prep_config, 
                         napari_viewer, input_volume_layer, save_dir, name, unet_or_config_file,
                         layer_reference, chunk_size, margin, debug)



# AUW - Config
# ------------

def affinity_watershed_prep_config(
        input_volume_layer,
        unet_or_config_file, 
        reference_layer
        ):
    # No config or unet path provided
    # -------------------------------
    if unet_or_config_file is None:
        unet = None
        affinities_extent = 1
    # Pathlib object though magicgui
    # ------------------------------
    if isinstance(unet_or_config_file, pathlib.PurePath):
        unet_or_config_file = str(unet_or_config_file)
    if isinstance(unet_or_config_file, str):
        affinities_extent = 1
        # Path provided
        # -------------
        if unet_or_config_file.endswith('.json'):
            config = read_config_json(unet_or_config_file)
            if config.get('unet') is None:
                unet = None
            if config.get('affinities_extent') is None:
                affinities_extent = 1
            if unet == 'labels layer':
                unet = reference_layer.metadata['unet']
            if unet == 'default':
                unet = None
        elif unet_or_config_file.endswith('.pt') or unet_or_config_file.endswith('.pth'):
            unet = unet_or_config_file
        # if a path is provided, 
    if unet is not None:
        if isinstance(unet, str):
            m = 'There was not file at the provided location: {unet}\nMake sure a pytorch unet lives here...'
            assert os.path.exists(unet), m
            assert unet.endswith('.pt') or unet.endswith('.pth')
        else:
            raise ValueError('Please provide a valid path to a pytorch unet - must end with .pt or .pth')
    
    # Output volume
    # -------------
    num_pred_channels = 3 * affinities_extent + 2
    output_volume = a_w_output_volume(input_volume_layer.data, num_pred_channels)

    # Load unet
    # ---------
    unet = load_unet(unet)
    
    # key word arguments for segmentation
    # -----------------------------------
    num_pred_channels = 3 * affinities_extent + 2
    new_config = {
        'unet': unet, 
        'output_volume' : output_volume, 
    }
    return new_config


def a_w_output_volume(data, num_pred_channels, **kwargs):
    output_volume = np.zeros((num_pred_channels,) + data.shape[-3:], dtype=np.float32) 
    return output_volume



# AUW - Processing function
# -------------------------

def affinity_watershed_for_chunks(
        input_volume, 
        current_output, 
        chunk_size, 
        margin,
        # specific
        unet=None,
        output_volume=None,
        **kwargs
        ):
    '''
    Function that inserts a the results of an affinity watershed into 
    current output

    Parameters
    ----------
    input_volume: array 
        This is the image to process
    output_volume: array
        This is the array that is used for intermediate processing
        steps (e.g., unet-derived feature map, LoG processed image). 
        In this case, it has five channels for the output of
        the affinity unet (z-1, y-1, x-1, mask, centre-points). 
    chunk_size: tuple
        This tells you how big the chunks to be processed are
    margin: tuple
        This tells you how much overlap between chunks is used
    current_output: array
        This is the 3D array that stores the labels for the particular
        slice. The labels are then written into the labels layer data, 
        which is then displayed on the viewer. 
    unet: str or None
        Path to the unet to load to process the data. This will come
        from the config file. If None, the default unet is used. 
    use_default_unet: bool
        Do you want to get a mask from the default unet (which has been
        trained to find platelets). 
    '''
    config = {'unet' : unet, 'output_volume' : output_volume}
    if output_volume is None:
        raise ValueError('output_volume must not be None. Please ensure the output volume is supplied in the config dict')
    if unet is None:
        raise ValueError('unet must not be none. Please ensure a unet was loaded and added to the config dict in the config function')
    process_chunks(input_volume, chunk_size, output_volume, margin, 
                   predict_chunk_feature_map, config=config)
    ws.segment_output_image(output_volume, affinities_channels=(0, 1, 2), 
                            thresholding_channel=3, centroids_channel=4,
                            out=current_output.ravel())
    output_volume[:] = 0


# ----------
# U-net Mask
# ----------

def unet_mask(
        napari_viewer, 
        input_volume_layer: napari.layers.Image,
        save_dir: Union[str, None] = None,
        name: str = 'labels-prediction',
        unet_or_config_file: Union[str, None] = None,
        layer_reference: Union[str, None] = None,
        chunk_size: tuple = (10, 256, 256),
        margin: tuple = (1, 64, 64),
    ):
    '''
    ...

    Parameters
    ----------
    napari_viewer: napari.Viewer
        The viewer into which the resultant segmentation will be placed.
    input_volume_layer: napari.layers.Image
        The viewer layer with the image to be processed.
    save_dir: str or None (None)
        The directory into which to save the outputted segmentation. 
    name: str ('labels-prediction')
        The name with which to save the data. Also the name of the 
        resultant segmentation layer that will be displayed in the 
        viewer. 
    config_file: str or None (None)
        Path to a json file with any additional parameters. I'm sure
        There is  a more elegant way to do this through the GUI, 
        but I don't have the time to work on this for now. 
    layer_reference: str or None (None)
        If you need information from one of the layers, this is the
        layer to get it from. This is fed into the custom config function. 
        This function can get a unet from layer metadata. 
    chunk_size: tuple (10, 256, 256)
         This tells you how big the chunks to be processed are
    margin: tuple (1, 64, 64)
        This tells you how much overlap between chunks is used
    '''
    segmentation_wrapper(unet_mask_for_chunks, unet_mask_prep_config, 
                         napari_viewer, input_volume_layer, save_dir, name, 
                         unet_or_config_file, layer_reference, chunk_size, margin)


# UM - Processing function
# ------------------------

def unet_mask_for_chunks(
        input_volume, 
        current_output, 
        chunk_size, 
        margin, 
        # specific
        output_volume,
        unet,
        **kwargs
        ):
    '''
    Function that inserts a unet-derived mask into current output.
    
    Parameters
    ----------
    input_volume: array 
        This is the image to process
    output_volume: array
        This is the array that is used for intermediate processing
        steps (e.g., unet-derived feature map, LoG processed image).
        In this case it is a feature map showing the probability that
        a pixel belongs to a segmentable object.
    chunk_size: tuple
        This tells you how big the chunks to be processed are
    margin: tuple
        This tells you how much overlap between chunks is used
    current_output: array
        This is the 3D array that stores the labels for the particular
        slice. The labels are then written into the labels layer data, 
        which is then displayed on the viewer/ 
    unet: str
        Path to the unet to load to process the data. This will come
        from the config file. 
    use_default_unet: bool
        Do you want to get a mask from the default unet (which has been
        trained to find platelets). There is a mask in channel 3. 
        This may or may not be useful. Regardless, it is accessible 
        through the unet-mask.  
    '''
    default_only_mask = unet == 'default' or unet is None 
    config = {
        'unet' : unet, 
        'default_only_mask' : default_only_mask
    }
    process_chunks(input_volume, chunk_size, output_volume, margin, 
                   predict_chunk_feature_map, 
                   prep_kwarg_function=load_unet, config=config)
    mask = ws._get_mask(output_volume)
    current_output[:, ...] = mask


# UM - Config
# -----------

def unet_mask_prep_config(
        unet, 
        which_unet, 
        reference_layer, 
        config, 
    ):
    if which_unet == 'default' or which_unet is None:
        unet = None
    if which_unet == 'file':
        unet = unet
    elif which_unet == 'labels layer':
        unet = reference_layer.metadata['unet']




# ---------
# Otsu Mask 
# ---------

def otsu_mask(
        napari_viewer, 
        input_volume_layer: napari.layers.Image,
        save_dir: Union[str, None] = None,
        name: str = 'labels-prediction',
        config_file: Union[str, None] = None,
        layer_reference: Union[str, None] = None,
        chunk_size: tuple = (10, 256, 256),
        margin: tuple = (1, 64, 64),
    ):

    '''
    Funtion for finding a simple mask using Otsu thresholding

    Parameters
    ----------
    napari_viewer: napari.Viewer
        The viewer into which the resultant segmentation will be placed.
    input_volume_layer: napari.layers.Image
        The viewer layer with the image to be processed.
    save_dir: str or None (None)
        The directory into which to save the outputted segmentation. 
    name: str ('labels-prediction')
        The name with which to save the data. Also the name of the 
        resultant segmentation layer that will be displayed in the 
        viewer. 
    config_file: str or None (None)
        Path to a json file with any additional parameters. I'm sure
        There is  a more elegant way to do this through the GUI, 
        but I don't have the time to work on this for now. 
    layer_reference: str or None (None)
        If you need information from one of the layers, this is the
        layer to get it from. This is fed into the custom config function. 
        This function doesn't use this. 
    chunk_size: tuple (10, 256, 256)
         This tells you how big the chunks to be processed are
    margin: tuple (1, 64, 64)
        This tells you how much overlap between chunks is used
    '''
    segmentation_wrapper(otsu_mask_for_chunks, otsu_mask_prep_config, 
                         napari_viewer, input_volume_layer, save_dir, name, config_file,
                         layer_reference, chunk_size, margin)



def otsu_mask_for_chunks(
        input_volume, 
        current_output, 
        chunk_size, 
        margin,
        # specific
        gaus_sigma,
        **kwargs
        ):
    '''
    Function that inserts a the results of an affinity watershed into 
    current output

    Parameters
    ----------
    input_volume: array 
        This is the image to process
    output_volume: array
        This is the array that is used for intermediate processing
        steps (e.g., unet-derived feature map, LoG processed image). 
        In this case, it is a smoothed image (using a median filter
        because I like to live dangerously). 
    chunk_size: tuple
        This tells you how big the chunks to be processed are
    margin: tuple
        This tells you how much overlap between chunks is used
    current_output: array
        This is the 3D array that stores the labels for the particular
        slice. The labels are then written into the labels layer data, 
        which is then displayed on the viewer. 
    '''
    mask = ws._get_mask(input_volume)
    current_output[:, ...] = mask


def otsu_mask_prep_config(gaus_sigma, **kwargs):
    pass




# --------------
# Blob Watershed
# --------------

def blob_watershed(
        napari_viewer, 
        input_volume_layer: napari.layers.Image,
        save_dir: Union[str, None] = None,
        name: str = 'labels-prediction',
        config_file: Union[str, None] = None,
        layer_reference: Union[str, None] = None,
        chunk_size: tuple = (10, 256, 256),
        margin: tuple = (1, 64, 64),
    ):
    '''
    Funtion for finding a simple mask using Otsu thresholding

    Parameters
    ----------
    napari_viewer: napari.Viewer
        The viewer into which the resultant segmentation will be placed.
    input_volume_layer: napari.layers.Image
        The viewer layer with the image to be processed.
    save_dir: str or None (None)
        The directory into which to save the outputted segmentation. 
    name: str ('labels-prediction')
        The name with which to save the data. Also the name of the 
        resultant segmentation layer that will be displayed in the 
        viewer. 
    config_file: str or None (None)
        Path to a json file with any additional parameters. I'm sure
        There is  a more elegant way to do this through the GUI, 
        but I don't have the time to work on this for now. 
    layer_reference: str or None (None)
        If you need information from one of the layers, this is the
        layer to get it from. This is fed into the custom config function. 
        This function doesn't use this. 
    chunk_size: tuple (10, 256, 256)
         This tells you how big the chunks to be processed are
    margin: tuple (1, 64, 64)
        This tells you how much overlap between chunks is used
    '''
    segmentation_wrapper(blob_watershed_for_chunks, blob_watershed_prep_config, 
                         napari_viewer, input_volume_layer, save_dir, name, config_file,
                         layer_reference, chunk_size, margin)


def blob_watershed_for_chunks(
        input_volume, 
        current_output, 
        chunk_size, 
        margin,
        # specific
        min_sigma,
        max_sigma, 
        num_sigma, 
        threshold, 
        gaus_sigma
        ):
    '''
    Function that inserts a the results of an affinity watershed into 
    current output

    Parameters
    ----------
    input_volume: array 
        This is the image to process
    output_volume: array
        This is the array that is used for intermediate processing
        steps (e.g., unet-derived feature map, LoG processed image). 
        In this case, it is a smoothed image (using a median filter
        because I like to live dangerously). 
    chunk_size: tuple
        This tells you how big the chunks to be processed are
    margin: tuple
        This tells you how much overlap between chunks is used
    current_output: array
        This is the 3D array that stores the labels for the particular
        slice. The labels are then written into the labels layer data, 
        which is then displayed on the viewer. 
    min_sigma: scalar or sequence of scalars
        Minimum sigma of a gaussian kernel to use for the laplacian 
        of gaussian (LoG) blob detection. The smaller this is, the smaller 
        the smallest objects you detect will be.
    max_sigma: scalar or sequence of scalars
        Maximum sigma of a gaussian kernel to use for the LoG blob 
        detection. The larger this is, the larger the smallest 
        objects you detect will be.
    num_sigma: int
        The number of intermediate values of standard deviations 
        to consider between min_sigma and max_sigma for LoG. 
    threshold: float or None
        The absolute lower bound for scale space maxima for LoG. 
        Local maxima smaller than threshold are ignored.
    gaus_sigma: scalar or sequence of scalars
        The sigma to use for the gaussian smoothing used to make
        the mask from the image.
    '''
    # find the blob seeds for the watershed
    markers = blob_log(input_volume, min_sigma=min_sigma, max_sigma=max_sigma, 
                         num_sigma=num_sigma, threshold=threshold)
    
    distance = ndi.distance_transform_edt(input_volume)
    mask = ws._get_mask(input_volume, gaus_sigma)
    labels = watershed(-distance, markers, mask=mask)
    current_output[:, ...] = labels
    

def blob_watershed_prep_config(
        max_sigma, 
        num_sigma, 
        threshold, 
        gaus_sigma, 
        **kwargs
        ):
    if min_sigma is None:
        min_sigma = 1
    if max_sigma is None:
        max_sigma = 30
    if num_sigma is None:
        num_sigma = 10
    if threshold is None:
        threshold = 0.1
    if gaus_sigma is None:
        gaus_sigma = 2
    new_kwargs = {
        'max_sigma' : max_sigma, 
        'min_sigma' : min_sigma,
        'num_sigma' : num_sigma, 
        'threshold' : threshold, 
        'gaus_sigma' : gaus_sigma
    }
    return new_kwargs



# ------------------
# DoG Blob Watershed
# ------------------

def dog_blob_watershed(
        napari_viewer, 
        input_volume_layer: napari.layers.Image,
        save_dir: Union[str, None] = None,
        name: str = 'labels-prediction',
        config_file: Union[str, None] = None,
        layer_reference: Union[str, None] = None,
        chunk_size: tuple = (10, 256, 256),
        margin: tuple = (1, 64, 64),
        debug: bool =False
    ):
    '''
    Funtion for segmenting an image using DoG blob detection

    Parameters
    ----------
    napari_viewer: napari.Viewer
        The viewer into which the resultant segmentation will be placed.
    input_volume_layer: napari.layers.Image
        The viewer layer with the image to be processed.
    save_dir: str or None (None)
        The directory into which to save the outputted segmentation. 
    name: str ('labels-prediction')
        The name with which to save the data. Also the name of the 
        resultant segmentation layer that will be displayed in the 
        viewer. 
    config_file: str or None (None)
        Path to a json file with any additional parameters. I'm sure
        There is  a more elegant way to do this through the GUI, 
        but I don't have the time to work on this for now. 
    layer_reference: str or None (None)
        If you need information from one of the layers, this is the
        layer to get it from. This is fed into the custom config function. 
        This function doesn't use this. 
    chunk_size: tuple (10, 256, 256)
         This tells you how big the chunks to be processed are
    margin: tuple (1, 64, 64)
        This tells you how much overlap between chunks is used
    '''
    segmentation_wrapper(dog_blob_watershed_for_chunks, dog_blob_watershed_prep_config, 
                         napari_viewer, input_volume_layer, save_dir, name, config_file,
                         layer_reference, chunk_size, margin, debug)


def dog_blob_watershed_for_chunks(
        input_volume, 
        current_output, 
        chunk_size, 
        margin,
        # specific
        min_sigma,
        max_sigma, 
        threshold, 
        **kwargs
        ):
    '''
    Function that inserts the results of a DoG blob detection into
    current_output.

    Parameters
    ----------
    input_volume: array 
        This is the image to process
    chunk_size: tuple
        Not used here.
    margin: tuple
        Not used here. 
    current_output: array
        This is the 3D array that stores the labels for the particular
        slice. The labels are then written into the labels layer data, 
        which is then displayed on the viewer. 
    min_sigma: scalar or sequence of scalars
        Minimum sigma of a gaussian kernel to use for the difference 
        of gaussian (DoG) blob detection. The smaller this is, the smaller 
        the smallest objects you detect will be.
    max_sigma: scalar or sequence of scalars
        Maximum sigma of a gaussian kernel to use for the DoG blob 
        detection. The larger this is, the larger the smallest 
        objects you detect will be.
    threshold: float or None
        The absolute lower bound for scale space maxima for LoG. 
        Local maxima smaller than threshold are ignored.
    gaus_sigma: scalar or sequence of scalars
        The sigma to use for the gaussian smoothing used to make
        the mask from the image.
    '''
    # find the blob seeds for the watershed
    input_volume = np.pad(input_volume, pad_width=1)
    dog = dog_image(input_volume, min_sigma, max_sigma)
    mask = dog > threshold
    markers = blob_dog(input_volume, min_sigma=min_sigma, max_sigma=max_sigma, 
                        threshold=threshold)
    distance = ndi.distance_transform_edt(input_volume)
    #local_maxi = peak_local_max(distance, min_distance=3, labels=input_volume)
    centroids = np.zeros(distance.shape, dtype=bool)
    idx = tuple(markers.T.astype(int))[:-1] # there's an extra index
    centroids[idx] = True
    markers, num_objects = ndi.label(centroids) #, structure=np.ones((3,3,3)))
    labels = watershed(-distance, markers, mask=mask)
    #distance = ndi.distance_transform_edt(input_volume)
    #mask = ws._get_mask(input_volume, gaus_sigma)
    #labels = watershed(-distance, markers, mask=mask)
    current_output[:, ...] = labels
    


def dog_blob_watershed_prep_config(
        input_volume_layer, # not used here
        unet_or_config_file, 
        reference_layer, # not used here
        max_sigma=1.5, 
        min_sigma=1,
        threshold=0.02, 
        ):
    if unet_or_config_file is not None:
        config = read_config_json(unet_or_config_file)
        val = config.get['max_sigma']
        max_sigma = val if val is not None else max_sigma
        val = config.get['min_sigma']
        min_sigma = val if val is not None else min_sigma
        val = config.get['threshold']
        threshold = val if val is not None else threshold
    new_kwargs = {
        'max_sigma' : max_sigma, 
        'min_sigma' : min_sigma,
        'threshold' : threshold, 
    }
    return new_kwargs


def dog_image(input_vol, sigma_min, sigma_max):
    dog = gaussian(input_vol, sigma_min) - gaussian(input_vol, sigma_max)
    return dog

# ---------------
# Common funcions
# ---------------


def read_config_json(path_to_json):
    with open(path_to_json, 'r') as f:
        data = json.load(f)
    return data


def output_vol_function_default(data):
    shape = (1, ) * (4 - data.ndim) + data.shape
    output_volume = np.zeros(data.shape, dtype=np.float32) 
    return output_volume



def segmentation_wrapper(
        processing_function: Callable, 
        config_prep_function: Callable,
        napari_viewer, 
        input_volume_layer: napari.layers.Image,
        save_dir: Union[str, None],
        name: str,
        network_or_config_file: Union[str, None],
        layer_reference: Union[str, None],
        chunk_size: tuple,
        margin: tuple,
        debug: bool =False
        ):
    """
    Function that takes a function for processing, preparing the configuration
    dict (key word arguments). 

    Parameters
    ----------
    processing_function: Callable
        The function you want to use to process the data. This function
        should take key word arguments (**kwargs). The function must
        take the following arguments in this order:
            - input_volume: array (3D) - the image frame to be processed
            - current_output: array (3D) - write the labels for the current frame here
            - chunk_size: tuple (3D) - size of chunks  
            - margin: tuple (3D) - size of margin
    config_prep_function: Callable (None)
        If not None, this function will be called before entering the 
        segmentation loop. It prepares a dictionary to be fed to the 
        process functioning as kew word arguments (**kwargs) when it
        is executed. 
            - input_volume_layer: napari.layers.Image - image layer to be processed
            - network_or_config_file: str - where to find the file you want to use
            - layer_reference: napari.layers.Layer - layer with metadata of interest
    napari_viewer: napari.Viewer
        The napari viewer. 
    input_volume_layer: napari.layers.Image
        The imput layer you want to segment. 
    save_dir: str or None
        The directory to which you want to save the segmentation to. 
    name: str
        The name you want to save the segmentation under. The segmentation
        will automatically be saved as a zarr with the extention .zarr. You 
        don't need to include an extension in the name. This name will also
        be used as the layer name. 
    network_or_config_file: str or None
        This is the file containing a neural network of 
    layer_reference: str or None
    chunk_size: tuple
    margin: tuple
    output_vol_function: function or None
    """
    
    viewer = napari_viewer

    # get the config dict
    # -------------------
    config = config_prep_function(input_volume_layer, network_or_config_file, layer_reference)
    if config is None:
        config = {}

    # Prepare data
    # ------------
    shape_4D = np.broadcast_shapes((1,) * 4, input_volume_layer.data.shape)
    data = np.reshape(input_volume_layer.data, shape_4D)
    scale = viewer.layers[0].scale[-3:] # lazy assumption that all layers have the same scale 
    translate = viewer.layers[0].translate[-3:] # and same translate 
    ndim = len(chunk_size)

    # Output labels
    # --------------
    # get the output labels and layer for output segmentation
    output_labels = zarr.zeros(
        shape=shape_4D, 
        chunks=(1,) + data.shape[-3:], 
        dtype=np.int32, 
        )
    output_layer = viewer.add_labels(
            output_labels,
            name=name,
            scale=scale,
            translate=translate,
            )

    max_t = data.shape[0] - 1

    # for yeilds
    # ----------
    def handle_yields(yielded_val):
        viewer.dims.current_step = (yielded_val, 0, 0, 0)
        print(f"Segmented t = {yielded_val}")
        if yielded_val == max_t and save_dir is not None:
            save_path = os.path.join(str(save_dir), name + '.zarr')
            zarr.save(save_path, output_labels)


    # for errors
    # ----------
    def handle_errors(errored_val):
        raise errored_val
    

    # define the worker
    # -----------------
    launch_worker = thread_worker(
        segmentation_loop,
        progress={'total': data.shape[0], 'desc': 'thread-progress'},
        # this does not preclude us from connecting other functions to any of the
        # worker signals (including `yielded`)
        connect={'yielded': handle_yields, 'errored': handle_errors},
    )

    # Launch the worker
    # -----------------
    if not debug:
        worker = launch_worker( # this is where the process dies... "There appear to be 2 leaked semaphore objects ..."
            viewer, 
            data, 
            chunk_size, 
            margin, 
            ndim, 
            output_labels, 
            processing_function,
            config
        )
    else:
        for t in segmentation_loop(viewer, data, chunk_size, margin, ndim, output_labels, processing_function, config):
            print(f'Segmented frame {t}')

    # Save the data
    # -------------
    if save_dir is not None and not debug:
        save_path = os.path.join(str(save_dir), name + '.zarr')
        zarr.save(save_path, output_labels)

    return output_layer


def segmentation_loop(
        viewer,
        data, 
        chunk_size, 
        margin, 
        ndim, 
        output_labels, 
        processing_function,
        config
    ):
    '''
    This function runs the segmentation function on one 3D frame at a time. 
    The function yeilds the current time point when done and this is printed
    in the main thread. 

    Parameters
    ----------
    data: np.array or zarr.array
    viewer: napari.Viewer
    output_volume: np.array or zarr.array
    chunk_size: tuple of len ndim
    margin: tuple of len ndim
    ndim: int
    output_labels: np.array or zarr.array
    processing_function: func
        Function for producing the full segmentation
    config: dict
        Key word arguments for processing function
    '''
    for t in range(data.shape[0]):
        #print(t) 
        slicing = (t, slice(None), slice(None), slice(None))
        input_volume = np.asarray(data[slicing]).astype(np.float32)
        if input_volume.min() == 0:
            input_volume = remove_sum_zero_slices(input_volume)
            #random_vol = np.random.normal(input_volume.mean(), size=input_volume.shape)
            #input_volume = np.where(input_volume == 0, random_vol, input_volume).astype(np.float32)
        input_volume /= np.max(input_volume)
        current_output = np.pad(
            np.zeros(data.shape[1:], dtype=np.uint32), # not sure if this will work
            1,
            mode='constant',
            constant_values=0,
            )
        crop = tuple([slice(1, -1),] * ndim)  # yapf: disable
        # predict using unet
        processing_function(input_volume, current_output, chunk_size, margin, **config)
        #current_output[crop] = np.where(input_volume == 0, 0, current_output[crop])
        output_labels[t, ...] = current_output[crop]
        yield t


def remove_sum_zero_slices(input_volume):
    for ax_i in range(input_volume.ndim):
            ax_nonzero_rows = []
            for i in range(input_volume.shape[ax_i]):
                s = [slice(None) for i in range(input_volume.ndim)]
                s[ax_i] = slice(i, i+1)
                s = tuple(s)
                if input_volume[s].sum() != 0:
                    ax_nonzero_rows.append(i)
            s = [slice(None) for i in range(input_volume.ndim)]
            s[ax_i] = ax_nonzero_rows
            s = tuple(s)
            input_volume = input_volume[s]
    return input_volume


# ------------
# summary dict 
# ------------

# to be used in segment_data doc widget
segmenters = {
    'affinity-unet-watershed' : affinity_unet_watershed, 
  #  'unet-mask' : unet_mask, 
   # 'otsu-mask' : otsu_mask,
  #  'LoG-blob-watershed' : blob_watershed,
    'DoG-blob-watershed' : dog_blob_watershed
}

