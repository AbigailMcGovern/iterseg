from .augment import augment_images
import dask.array as da
from datetime import datetime
from .helpers import get_files, log_dir_or_None, write_log, LINE
from .labels import get_training_labels, print_labels_info
import numpy as np
import os
import pandas as pd
from pathlib import Path
import re
from tifffile import TiffWriter, imread
import torch 
from tqdm import tqdm
import zarr
import dask.array as da
from napari.types import ImageData, LabelsData


# -------------------
# Generate Train Data
# -------------------
def get_train_data(
    image_list, 
    gt_list,
    out_dir=None, 
    name='train-unet',
    shape=(10, 256, 256), 
    n_each=100, 
    channels=('z-1', 'y-1', 'x-1', 'centreness'),
    scale=(4, 1, 1), 
    log=True, 
    validation_prop=0.2,
):
    """
    Generate training data from whole ground truth volumes.

    Parameters
    ----------
    image_paths: str
        path to the image zarr file, which must be in tczyx dim format (ome bioformats)
    gt_paths: str
        path to the ground truth zarr file, which must be in tczyx dim format
    shape: tuple of int
        Shape of the test data to 
    channels: tuple of str
        tuple of channels to be added to the training data. 
            Affinities: 'axis-n' (pattern: r'[xyz]-\d+' e.g., 'z-1')
            Centreness: 'centreness'
    scale: tuple of numeric
        Scale of channels. This is used in calculating centreness score.
    log: bool
        Should print out be recorded in out_dir/log.txt?

    Returns
    -------
    chunk_dict: dict
        Dict containing generated training data of the form
        {
            'channels' : {
                'name' : (<channel name>, ...), 
                ...
            },
            'save_dir' : <path/to/random/chunks>,
            'labels_dirs' : {
                'name' : <path/to/random/chunks/labels>, 
                ...
            },
            'ids' : [<chunk id string>, ...]
            'x' : [<image chunk tensor>, ...], 
            'ground_truth' : [<gt chunk tensor>, ...], 
            'ys' : {
                'name'
            }
        }

    Notes
    -----
    if there is only one set of training labels generated, the name
    of the labels (anywhere you see 'name' above) will be 'y'.
    """
    assert len(image_list) == len(gt_list)
    now = datetime.now()
    d = now.strftime("%y%m%d_%H%M%S") + '_' + name
    out_dir = os.path.join(out_dir, d)
    os.makedirs(out_dir, exist_ok=True)
    chunk_dicts = []
    if not isinstance(scale, list):
        scale = [scale, ] * len(image_list)
    for i in range(len(image_list)):
        chunk_dict = get_random_chunks(
            image_list[i], 
            gt_list[i], 
            out_dir, 
            name=name,
            shape=shape, 
            n=n_each, 
            channels=channels,
            scale=scale[i], 
            log=log, 
            image_no=i
        )
        chunk_dicts.append(chunk_dict)
    chunk_dicts = concat_chunk_dicts(chunk_dicts)
    train_dicts = chunk_dict_to_train_dict(chunk_dicts, validation_prop)
    return train_dicts



def get_random_chunks(
                      image_src, 
                      gt_src, 
                      out_dir, 
                      name='unet-training',
                      shape=(10, 256, 256), 
                      n=25, 
                      min_brightness_prop=0.005, 
                      channels=('z-1', 'y-1', 'x-1', 'centreness'),
                      scale=(4, 1, 1), 
                      log=True, 
                      image_no=0
                      ):
    '''
    Obtain random chunks of data from whole ground truth volumes.

    Parameters
    ----------
    image: array like
        Image from which to generate training chunks
    ground_truth: array like
        ground truth segmentation (same shape as image)
    shape: tuple of int
        shape of chunks to obtain
    n: int
        number of random chunks to obtain
    min_brightness_prop:
        minimum cut off sum of affinities for an image.
        As affinities as belong to {0, 1}, this param
        is the number of voxels that boarder labels.
    
    Returns
    -------
    chunk_dict:
 
    '''
    save_output = out_dir is not None
    now = datetime.now()
    d = now.strftime("%y%m%d_%H%M%S") + '_' + name
    if isinstance(image_src, str):
        image = zarr.open_array(image_src)
        im_name = image_src
    else:
        try:
            im_shape = image_src.shape
        except:
            m = f'input image should be a path (str), napari image layer, or array like not {type(image_src)}'
            raise TypeError(m)
        image = image_src
        im_name = f'image_shape-{im_shape}_prepared-{d}'
    image = normalise_data(np.array(image))
    if isinstance(gt_src, str):
        ground_truth = zarr.open_array(gt_src)
        gt_name = gt_src
    else:
        try:
            gt_shape = gt_src.shape
        except:
            m = f'input image should be a path (str), napari image layer, or array like not {type(gt_src)}'
            raise TypeError(m)
        gt_name = f'labels_shape-{gt_shape}_prepared-{d}'
        ground_truth = gt_src
    ground_truth = np.array(ground_truth)
    print(LINE)
    s = f'Generating training data from image: {im_name}, Ground truth: {gt_name}'
    print(s)
    # in the following code, image chunks are np.ndarray
    print('Generating random image chunks...')
    chunk_dict = get_image_chunks(
        image, 
        shape=shape, 
        n=n, 
        min_brightness_prop=min_brightness_prop, 
        image_no=image_no
    )
    chunk_dict['df']['image_no'] = [image_no, ] * len(chunk_dict['df'])
    chunk_dict['df']['image_file'] = [Path(im_name).stem, ] * len(chunk_dict['df'])
    print('Generating training labels...')
    chunk_dict = get_labels_chunks(chunk_dict, ground_truth, 
                                   channels=channels, scale=scale)
    print('Augmenting data...')
    chunk_dict = augment_chunks(chunk_dict)
    # save the chunks whilst np.ndarray
    save_dir = None
    if save_output:
        print('Saving for posterity...')
        save_dir = save_from_chunk_dict(chunk_dict, out_dir, name)
    # convert to tensor 
    #print('Converting to Torch Tensor...')
    #convert_chunks_to_tensor(chunk_dict)
        if log:
            write_log(LINE, save_dir)
            write_log(s, save_dir)
        print(LINE)
        s = f'Obtained {n} {shape} chunks of training data'
        print(s)
        if log:
            write_log(LINE, save_dir)
            write_log(s, save_dir)
    log_dir = log_dir_or_None(log, save_dir)
    print_labels_info(channels, out_dir=log_dir)
    df_path = os.path.join(save_dir, 'start_coords.csv')
    if os.path.exists(df_path):
        df = pd.read_csv(df_path)
        df = pd.concat([df, chunk_dict['df']])
    else:
        df = chunk_dict['df']
    df.to_csv(df_path)
    return chunk_dict


# -----------------------
# Get Random Image Chunks 
# -----------------------

def get_image_chunks(
    image, 
    shape=(10, 256, 256), 
    n=25, 
    min_brightness_prop=0.3, 
    image_no=0,
):
    im = np.array(image)
    assert len(im.shape) == len(shape)
    xs = []
    ids = []
    i = 0
    df = {'z_start' : [],
          'y_start' : [],
          'x_start' : []}
    slices = []
    while i < n:
        dim_randints = []
        for j, dim in enumerate(shape):
            max_ = im.shape[j] - dim - 1
            ri = np.random.randint(0, max_)
            dim_randints.append(ri)
        # Get the image chunk
        s_ = [slice(dim_randints[j], dim_randints[j] + shape[j]) for j in range(len(shape))]
        s_ = tuple(s_)
        x = im[s_]
        mbrightprop = x.mean() / x.max()
        if mbrightprop > min_brightness_prop: # if the image is bright enough
            # add the image slice to the slices dict (to be used later for GT segmentation)
            slices.append(s_)
            # add coords to output df
            for j in range(len(shape)):
                _add_to_dataframe(j, dim_randints[j], df)
            # add the image to the list of output chunks 
            # - note that these have yet to be augmented 
            xs.append(x)
            # get the datetime to give the samples unique names 
            now = datetime.now()
            d = now.strftime("%y%m%d_%H%M%S") + f'_img-{image_no}_chunk-{i}'
            ids.append(d)
            # another successful addition, job well done you crazy mofo
            i += 1
    df['data_ids'] = ids
    chunk_dict = {
        'x' : xs, 
        'slices' : slices, 
        'ids' : ids,
        'df' : df,
        'n' : len(xs)
    }
    chunk_dict['df'] = pd.DataFrame(chunk_dict['df'])
    return chunk_dict

# referenced in get_image_chunks(...)
def _add_to_dataframe(dim, start, df):
    if dim == 0:
        df['z_start'].append(start)
    if dim == 1:
        df['y_start'].append(start)
    if dim == 2:
        df['x_start'].append(start)


# ------------------------------
# Get Training Labels for Chunks
# ------------------------------

def get_labels_chunks(
    chunk_dict,
    ground_truth, 
    channels=('z-1', 'y-1', 'x-1', 'centreness-log'), 
    scale=(4, 1, 1), 
):  
    if not isinstance(channels, dict):
        channels = {'y' : channels}
    labels = {}
    #print(channels)
    chunk_dict['channels'] = channels
    for key in channels.keys():
        labs = get_training_labels(ground_truth, channels[key], scale)
        labels[key] = labs
    chunk_dict['ys'] = {}
    for key in labels.keys():
        chunk_dict['ys'][key] = []
    chunk_dict['ground_truth'] = []
    for s_ in chunk_dict['slices']:
        # add the segmentation chunks
        gt = ground_truth[s_]
        chunk_dict['ground_truth'].append(gt)
        # add the training labels chunks
        new_s_ = [slice(None, None), ] + list(s_)
        new_s_ = tuple(new_s_)
        for key in labels.keys():
            labs = labels[key]
            y = labs[new_s_]
            chunk_dict['ys'][key].append(y)
    return chunk_dict


# --------------
# Augment Chunks
# --------------
# Apply augmentation to each chunk. The augmentations are random and are 
# applied to each image, the ground truth, and the training labels. 
# If multiple sets of training labels are used, the same augmentation
# is applied to each. 

def augment_chunks(chunk_dict):
    x, ys, labs_keys, gt, n = _read_chunk_dict(chunk_dict)
    # augment each chunk overwriting memory as you go
    for i in range(n):
        labels_dict = {key: ys[key][i] for key in labs_keys}
        image, labels_dict, ground_truth = augment_images(x[i], labels_dict, gt[i])
        chunk_dict['x'][i] = image
        for key in labs_keys:
            chunk_dict['ys'][key][i] = labels_dict[key]
        chunk_dict['ground_truth'][i] = ground_truth
    return chunk_dict


def _read_chunk_dict(chunk_dict):
    x = chunk_dict['x'] # list (len n)
    ys = chunk_dict['ys'] # dict of lists (len n)
    labs_keys = list(ys.keys())
    gt = chunk_dict['ground_truth'] # list (len n)
    # check that all lists are len n
    n = chunk_dict['n']
    assert n == len(x)
    assert n == len(gt)
    for key in ys.keys():
        assert len(ys[key]) == n
    return x, ys, labs_keys, gt, n


# -----------
# Save Output
# -----------
# Saved output may be loaded at a later time for training (perhaps using 
# different trainin conditions), for reference, or used to test/visualise 
# quality of results. Saved output is generally not required for training 
# itself. 

def save_from_chunk_dict(chunk_dict, out_dir, name):
    x = chunk_dict['x'] # list (len n)
    ys = chunk_dict['ys'] # dict of lists (len n)
    labs_keys = list(ys.keys())
    channels = chunk_dict['channels']
    gt = chunk_dict['ground_truth'] # list (len n)
    ids = chunk_dict['ids']
    # get the folder to which to save the data
    chunk_dict['name'] = name
    # save the images and gt
    for i in range(len(x)):
        save_chunk(out_dir, i, x, ids, '_image.zarr')
        save_chunk(out_dir, i, gt, ids, '_GT.zarr')
    # make subdirectories for training labels and save
    labs_paths = {}
    for key in labs_keys:
        chans = channels[key]
        i = 0
        path = os.path.join(out_dir, str(key))
        labs_paths[key] = path
        os.makedirs(path, exist_ok=True)
        y = ys[key]
        for j in range(len(y)):
            save_chunk(path, i, y, ids, '_labels.zarr')
        i += 1
    chunk_dict['save_dir'] = out_dir
    chunk_dict['labels_dirs'] = labs_paths
    # convert ground truth to lazy rep so that it is accessible for vis 
    # but not using memory
    chunk_dict['ground_truth'] = da.stack(
        [da.from_array(chunk_dict['ground_truth'][i], 
         chunks=chunk_dict['ground_truth'][i].shape) \
        for i in range(len(chunk_dict['ground_truth']))])
    return out_dir
        

# -----------------------------
# Dask, Numpy, Torch Conversion
# -----------------------------

def convert_chunks_to_tensor(chunk_dict):
    x, ys, labs_keys, gt, n = _read_chunk_dict(chunk_dict)
    for i in range(n):
        chunk_dict['x'][i] = torch.from_numpy(x[i].copy())
        for key in labs_keys:
            chunk_dict['ys'][key][i] = torch.from_numpy(ys[key][i].copy())
        # chunk_dict['ground_truth'] = torch.from_numpy(gt[i].copy())
    return chunk_dict


def train_chunks_to_dask(train_dict):
    convert = ['x', 'vx', 'y', 'vy']
    for key in convert:
        stack = np.stack(train_dict[key])
        stack = da.from_array(stack, chunks=(1, ) + stack.shape[1:])
        train_dict[key] = stack
    return train_dict


def load_dask_as_tensor(i, stack):
    chunk = stack[i, ...].compute()
    chunk = torch.from_numpy(chunk)
    return chunk


def load_tensor_from_numpy(i, ls):
    chunk = ls[i].copy()
    chunk = torch.from_numpy(chunk)
    return chunk


def load_tensor_from_zarr(i, ls):
    chunk = np.array(ls[i])
    chunk = torch.from_numpy(chunk)
    return chunk

# ------------------------
# Concatenate chunk_dicts
# ------------------------

def concat_chunk_dicts(chunks_dict_list):
    # assume that the dicts have the right structure
    c = 0
    gt = []
    for chunk_dict in chunks_dict_list:
        if c == 0:
            full_dict = chunk_dict
            gt.append(chunk_dict['ground_truth'])
            c += 1
        else:
            full_dict['x'] = full_dict['x'] + chunk_dict['x']
            gt.append(chunk_dict['ground_truth'])
            full_dict['ids'] = full_dict['ids'] + chunk_dict['ids']
            for key in full_dict['ys'].keys():
                full_dict['ys'][key] = full_dict['ys'][key] + chunk_dict['ys'][key]
            full_dict['df'] = pd.concat([full_dict['df'], chunk_dict['df']])
            full_dict['n'] = full_dict['n'] + chunk_dict['n']
    gt = da.concatenate(gt, axis=0)
    full_dict['ground_truth'] = gt
    return full_dict




# ---------------------
# Convert to train_dict
# ---------------------

def chunk_dict_to_train_dict(chunk_dict, validation_prop=0.2):
    '''
    Will convert a chunk_dict into as many train_dicts as there
    are sets of labels to train on. 
    '''
    n = len(chunk_dict['x'])
    no_val = np.round(validation_prop * n).astype(int)
    vx_idx = np.random.randint(0, n, size=no_val)
    out = {}
    for key in chunk_dict['ys'].keys():
        train_dict = {
            'x' : [x for i, x in enumerate(chunk_dict['x']) if i not in vx_idx], 
            'vx' : [x for i, x in enumerate(chunk_dict['x']) if i in vx_idx], 
            'y' : [y for i, y in enumerate(chunk_dict['ys'][key]) if i not in vx_idx],
            'vy' : [y for i, y in enumerate(chunk_dict['ys'][key]) if i in vx_idx], 
            'ids' : [ID for i, ID in enumerate(chunk_dict['ids']) if i not in vx_idx],
            'vids' : [ID for i, ID in enumerate(chunk_dict['ids']) if i in vx_idx], 
            'out_dir' : chunk_dict['labels_dirs'][key], 
            'name' : key, 
            'channels' : chunk_dict['channels'][key]
        }
        #train_dict = train_chunks_to_dask(train_dict)
        print(f'generated train dict for {key}')
        out[key] = train_dict
    return out


# ------------------------
# General Helper Functions
# ------------------------

def normalise_data(image):
    '''
    Bring image values to between 0-1

    Parameters
    ----------
    image: np.array
        Image data. Dtype should be float.
    '''
    im = image / image.max()
    return im


def save_chunk(out_dir, i, data_list, ID_list, type_suffix):
    ID = ID_list[i]
    data = data_list[i]
    name = ID + type_suffix
    path = os.path.join(out_dir, name)
    zarr.save(path, data)
    # want to use out of memory zarr rather than in memory array
    arr = zarr.open_array(path, 'r')
    data_list[i] = arr


# -----------------------------------------------------------------------------
# Load Train Data
# -----------------------------------------------------------------------------

def load_train_data(
    data_dir, 
    id_regex=r'\d{6}_\d{6}_\d{1,3}',
    x_suffix='_image.tif', 
    y_suffix='_labels.tif',
    gt_suffix='_GT.tif',
    load_GT=False
    ):
    pass


def load_train_data(
                    data_dir, 
                    id_regex=r'\d{6}_\d{6}_\d{1,3}',
                    x_regex=r'\d{6}_\d{6}_\d{1,3}_image.tif', 
                    y_regex=r'\d{6}_\d{6}_\d{1,3}_labels.tif'
                    ):
    '''
    Load train data from a directory according to a naming convention

    Parameters
    ----------
    data_dir: str
        Directory containing data
    id_regex: r string
        regex that will be used to extract IDs that will
        be used to label network output
    x_regex: r string
        regex that represents image file names, complete with
        extension (tiff please)
    y_regex: r string
        regex that represents training label files, complete
        with extension (tiff please)

    Returns
    -------
    xs: list of torch.Tensor
        List of images for training 
    ys: list of torch.Tensor
        List of affinities for training
    ids: list of str
        ID strings by which each image and label are named.
        Eventually used for correctly labeling network output
    
    '''
    # Get file names for images and training labels
    x_paths, y_paths = get_files(
                                 data_dir, 
                                 x_regex=x_regex, 
                                 y_regex=x_regex
                                 )
    # Get IDs 
    id_pattern = re.compile(id_regex)
    ids = []
    x_paths.sort()
    y_paths.sort()
    for i in range(len(x_paths)):
        xn = Path(x_paths[i]).stem # this could have been avoided
        yn = Path(y_paths[i]).stem # why would I bother now though?!
        # assumes there will be a match for each
        xid = id_pattern.search(xn)[0]
        yid = id_pattern.search(yn)[0]
        m = 'There is a mismatch in image and label IDs'
        assert xid == yid, m
        ids.append(xid)
    # Get images and training labels in tensor form 
    xs = []
    ys = []
    for i in range(len(x_paths)):
        xp = x_paths[i]
        yp = y_paths[i]
        x = imread(xp)
        x = normalise_data(x)
        y = imread(yp)
        xs.append(torch.from_numpy(x))
        ys.append(torch.from_numpy(y))
    # returns objects in the same manner as get_train_data()
    print('------------------------------------------------------------')
    print(f'Loaded {len(xs)} sets of training data')
    print_labels_info(ys[0].shape)
    return xs, ys, ids


if __name__ =="__main__":
    import zarr
    import napari
    # Directory for training data and network output 
    #data_dir = '/Users/amcg0011/Data/pia-tracking/cang_training'
    data_dir = '/home/abigail/data/platelet-segmentation-training'
    # Path for original image volumes for which GT was generated
    image_paths = [os.path.join(data_dir, '191113_IVMTR26_I3_E3_t58_cang_training_image.zarr')] 
    # Path for GT labels volumes
    labels_paths = [os.path.join(data_dir, '191113_IVMTR26_I3_E3_t58_cang_training_labels.zarr')]
    labs = zarr.open(labels_paths[0])
    labs = np.array(labs)
    #cent_off = get_centre_offsets(labs, (4, 1, 1))
    #v = napari.Viewer()
    #z = cent_off[0] # - cent_off[0].min()
    #v.add_image(z, name='Z offsets', colormap='bop purple', blending='additive', scale=(4, 1, 1))
    #y = cent_off[1] #- cent_off[1].min()
    #v.add_image(y, name='Y offsets', colormap='bop orange', blending='additive', scale=(4, 1, 1))
    #x = cent_off[2] #- cent_off[2].min()
    #v.add_image(x, name='X offsets', colormap='bop blue', blending='additive', scale=(4, 1, 1))
    #napari.run()
