from iterseg.train_io import get_train_data
from iterseg.training_experiments import get_experiment_dict
from iterseg._dock_widgets import construct_channels_list, construct_conditions_list
import numpy as np
import os
from pathlib import Path
import zarr

FILE_PATH = __file__
ITERSEG_PATH = Path(FILE_PATH).parents[1]
DATA_PATH = os.path.join(ITERSEG_PATH, 'data')
OUT_PATH = os.path.join(DATA_PATH, 'temp')
gt_dir = os.path.join(DATA_PATH, 'GT_in_frames')
gt_files = [os.path.join(gt_dir, '191016_IVMTR12_Inj4_cang_exp3_fr125_GT_PL.zarr'), 
            os.path.join(gt_dir, '210511_IVMTR105_Inj5_DMSO2_exp3_fr54_GT_PL.zarr')]

images = [np.random.random((33, 512, 512)) for _ in range(2)]
ground_truth = [np.array(zarr.open(f)) for f in gt_files]



def test_z1_y1_x1_m_cl():
    #scale = [(4, 1, 1), (4, 1, 1)]
    n_each = 10
    channels = ('z-1', 'y-1', 'x-1', 'mask', 'centreness-log')
    gtd_kwargs = {
        'validation_prop': 0.2, 
        'n_each': n_each, 
        'scale': (4, 1, 1), 
        'name': 'train-unet', 
        'channels': {'my-unet': channels }
        }
    train_dict = get_train_data(images, ground_truth, out_dir=OUT_PATH, **gtd_kwargs)


def test_z1_y1_x1_m_c():
    #scale = [(4, 1, 1), (4, 1, 1)]
    n_each = 10
    channels = ('z-1', 'y-1', 'x-1', 'mask', 'centreness')
    gtd_kwargs = {
        'validation_prop': 0.2, 
        'n_each': n_each, 
        'scale': (4, 1, 1), 
        'name': 'train-unet', 
        'channels': {'my-unet': channels }
        }
    train_dict = get_train_data(images, ground_truth, out_dir=OUT_PATH, **gtd_kwargs)


def test_z1_y1_x1_m_cg():
    #scale = [(4, 1, 1), (4, 1, 1)]
    n_each = 10
    channels = ('z-1', 'y-1', 'x-1', 'mask', 'centroid-gauss')
    gtd_kwargs = {
        'validation_prop': 0.2, 
        'n_each': n_each, 
        'scale': (4, 1, 1), 
        'name': 'train-unet', 
        'channels': {'my-unet': channels }
        }
    train_dict = get_train_data(images, ground_truth, out_dir=OUT_PATH, **gtd_kwargs)


def test_get_exp_dict():
    condition_list = construct_conditions_list(images, 'BCELoss', 0.01, 2, (4, 1, 1))
    channels_list = construct_channels_list(1, 'mask', 'centreness')
    experiment_dict = get_experiment_dict(channels_list, ['default', ], 
                                          conditions_list=condition_list, name='train-unet', 
                                          validation_prop=0.2, n_each=100, scale=(4, 1, 1))



#test_z1_y1_x1_m_cl()
#test_z1_y1_x1_m_c()
#test_z1_y1_x1_m_cg()
#test_get_exp_dict()


# -----------------------
# Some examples of output
# -----------------------

# construct_channels_list
chl = [('z-1', 'y-1', 'x-1', 'mask', 'centreness-log')]
# construct_conditions_list
col = [{'scale': 
            [(1, 1, 1), (1, 1, 1), (1, 1, 1), 
             (1, 1, 1), (1, 1, 1), (1, 1, 1), 
             (1, 1, 1)], 
       'lr': 0.01, 
       'loss_function': 'BCELoss', 
       'epochs': 4}]
# experiment_dict
ed = {
    'get_train_data': 
        {
        'validation_prop': 0.2, 
        'n_each': 50, 
        'scale': (4, 1, 1), 
        'name': 'train-unet', 
        'channels': 
            {
            'my-unet': ('z-1', 'y-1', 'x-1', 'mask', 'centreness-log')
            }
        }, 
    'my-unet': 
        {
            'scale': [(1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1)], 
            'epochs': 4, 
            'lr': 0.01, 
            'loss_function': 'BCELoss', 
            'chan_weights': None, 
            'weights': None, 
            'update_every': 20, 
            'fork_channels': None
        }
    }