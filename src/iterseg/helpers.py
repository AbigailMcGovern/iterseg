import dask.array as da
from dask import delayed
import numpy as np
import os
from pathlib import Path
import re
import skimage.io as io


LINE = '------------------------------------------------------------'

def get_files(
              data_dir, 
              x_regex=r'\d{6}_\d{6}_\d{1,3}_image.tif', 
              y_regex=r'\d{6}_\d{6}_\d{1,3}_labels.tif'
              ):
    '''
    '''
    files = os.listdir(data_dir)
    x_paths = get_paths(
                        data_dir, 
                        regex=x_regex, 
                        )
    y_paths = get_paths(
                        data_dir, 
                        regex=y_regex, 
                        )
    m = 'There is a mismatch in the number of images and training labels'
    assert len(x_paths) == len(y_paths), m
    return x_paths, y_paths


def get_paths(
              data_dir, 
              regex=r'\d{6}_\d{6}_\d{1,3}_output.tif'
              ):
    '''
    Awesome default for regex... just so lazy !!
    '''
    files = os.listdir(data_dir)
    paths = []
    pattern = re.compile(regex)
    for f in files:
        match = pattern.search(f)
        if match is not None:
            paths.append(os.path.join(data_dir, match[0]))
    return paths


def write_log(string, out_dir, log_name='log.txt'):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, log_name), 'a') as log:
        log.write(string + '\n')


def log_dir_or_None(log, out_dir):
    if log:
        return out_dir
    else:
        return None

def get_ids(paths, regex=r'\d{6}_\d{6}_\d{1,3}'):
    ids = []
    pattern = re.compile(regex)
    for p in paths:
        p = Path(p)
        name = p.stem
        match = pattern.search(name)
        if match is not None:
            ids.append(match[0])
        else:
            raise ValueError('Irregular ID for training data file: must be YYMMDD_HHMMSS_<digit>')
    return ids


def check_ids_match(x, y, regex=r'\d{6}_\d{6}_\d{1,3}'):
    pattern = re.compile(regex)
    m = f'Number of '
    assert len(x) == len(y), m
    for i in range(len(x)):
        if not os.path.exists(x[i]): # if just an id string
            assert x[i] == y[i]
        else: # if the ids to be matched are in path stems
            xn = Path(x).stem
            yn = Path(y).stem
            xid = pattern.search(xn)[0] # will raise error if no id found
            yid = pattern.search(yn)[0]
            assert xid == yid


# -------------------------
# Load Output from Training 
# -------------------------

def get_dataset(train_dir, out_dir=None, GT=False, validation=False, return_ID=False):
    # directory for output, if none, assume output is with training data
    if out_dir is None:
        out_dir = train_dir
    # Get output regular expression
    if validation == False:
        out_regex = r'\d{6}_\d{6}_\d{1,3}_output.tif'
        o_s = '_output.tif'
    else:
        out_regex = r'\d{6}_\d{6}_\d{1,3}_validation_output.tif'
        o_s = '_validation_output.tif'
    # Get output paths and IDs
    output_paths = sorted(get_paths(out_dir, out_regex))
    ids = get_ids(output_paths)
    #output = get_regex_images(out_dir, out_regex, ids)
    # get training data (images and labels) according to id strings, 
    #   which correspond to the batch number.
    suffixes = ['_image.tif', '_labels.tif', o_s]
    if GT:
        suffixes.append('_GT.tif')
        labs, images, output, ground_truth =  get_data_by_id(train_dir, suffixes, 
                                       out_dir=out_dir, validation=validation)
        if not return_ID:
            return images, labs, output, ground_truth
        else:
            return images, labs, output, ground_truth, ids
    else:
        labs, images, output =  get_data_by_id(train_dir, suffixes, 
                                       out_dir=out_dir, validation=validation)
        if not return_ID:
            return images, labs, output
        else:
            return images, labs, output, ids


def get_dataset_segs(train_dir, out_dir=None, validation=True):
    suffixes = ('_GT.tif', '_segmentation.tif', '_DoG-segmentation.tif', '_image.tif')
    gt, seg, dog, im = get_data_by_id(train_dir, suffixes, 
                              out_dir=out_dir, validation=validation)
    return gt, seg, dog, im


def get_data_by_id(train_dir, suffixes, out_dir=None, validation=False):
    # directory for output, if none, assume output is with training data
    if out_dir is None:
        out_dir = train_dir
    # Get output regular expression
    if not validation:
        out_regex = r'\d{6}_\d{6}_\d{1,3}_output.tif'
    else:
        out_regex = r'\d{6}_\d{6}_\d{1,3}_validation_output.tif'
    # Get output paths and IDs
    output_paths = sorted(get_paths(out_dir, out_regex))
    ids = get_ids(output_paths)
    id_regex = r'\d{6}_\d{6}_\d{1,3}'
    out = []
    for s in suffixes:
        regex = id_regex + s
        out.append(get_regex_images(train_dir, regex, ids))
    return tuple(out)


def get_regex_images(data_dir, regex, ids, id_regex=r'\d{6}_\d{6}_\d{1,3}'):
    id_pattern = re.compile(id_regex)
    imread = delayed(_imread_squeeze, pure=True)  
    file_paths = sorted(get_paths(data_dir, regex))
    correct_paths = []
    for ID in ids:
        id_done = False
        for f in file_paths:
            n = Path(f).stem
            id_match = id_pattern.search(n)[0] # assumes there will be one
            if id_match == ID:
                correct_paths.append(f)
                id_done = True
        m = f'No file match was found for ID: {ID}'
        assert id_done, m
    images = [imread(path) for path in correct_paths]  
    sample = images[0].compute()  
    arrays = [da.from_delayed(image,           
                              dtype=sample.dtype,   
                              shape=sample.shape)
                                for image in images]

    stack = da.stack(arrays, axis=0)
    return stack


def _imread_squeeze(path):
    img = io.imread(path)
    return np.squeeze(img)