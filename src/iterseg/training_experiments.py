from datetime import datetime
import os
from . import train
from .train_io import get_train_data
import torch.nn as nn
from pathlib import Path


def run_experiment(
    experiment_dict, 
    image_list, 
    labels_list, 
    out_dir, 
    *args, 
    **kwargs
    ):
    gtd_kwargs = experiment_dict['get_train_data']
    train_dict = get_train_data(image_list, labels_list, out_dir, **gtd_kwargs)
    unets = {}
    for key in train_dict.keys():
        train_kwargs = train_dict[key]
        train_kwargs.update(experiment_dict[key])
        unet, unet_path = train.train_unet(**train_kwargs)
        unets[key] = {'unet': unet, 'unet_path' : unet_path}
    upper_dir = Path(train_kwargs['out_dir']).parents[1]
    unet_path_log = upper_dir / 'unet_paths.txt'
    s = [unets[key]['unet_path'] for key in unets.keys()]
    with open(unet_path_log, 'a') as f:
        st = str(s)
        f.write(st)
    return s


#def train_experiment(
 #   experiments, 
  #  out_dir, 
   # image_paths, 
#    labels_path
 #   exp, 
  #  ):
   # exp_dir = os.path.join(out_dir, experiments[exp]['suffix'])
    #    unet = train.train_unet_get_labels(
     #           exp_dir, 
      #          image_paths, 
       #         labels_paths, 
        #        **experiments[exp]) 
       # exp['unet'] = unet


def get_experiment_dict(
    channels_list,
    condition_names,
    conditions_list=None,
    name='train-unet' ,
    validation_prop=0.2, 
    n_each=100,
    scale=(4, 1, 1),
    **kwargs
    ):
    '''
    Info about conditions to test. 

    Parameters
    ----------
    channels_list: list
        ...
    condition_names: list
        ...
    conditions_list: list
        ...
    name: str
    validation_prop: float
    n_each: int
    scale: tuple of float
    '''
    # get the kwargs for obtaining the training data
    experiment = {}
    experiment['get_train_data'] = {
       'validation_prop' : validation_prop, 
        'n_each' : n_each, 
        'scale' : scale, 
        'name' : name, 
        'channels' : {}
    }
    for i, nm in enumerate(condition_names):
        experiment['get_train_data']['channels'][nm] = channels_list[i]
    # get the kwargs for training under each condition
    for i in range(len(condition_names)):
        experiment[condition_names[i]] = {
            'scale' : scale,
            'epochs' : 4,
            'lr' : .01,
            'loss_function' : 'BCELoss',
            'chan_weights' : None, 
            'weights' : None,
            'update_every' : 20, 
            'fork_channels' : None
        }
        if conditions_list is not None:
            custom_kw = conditions_list[i]
            for key in custom_kw.keys():
                experiment[condition_names[i]][key] = custom_kw[key]
    if 'mask' in experiment['get_train_data']['channels']:
        experiment['get_train_data']['absolute_thresh'] = 0.5
    return experiment


# -----------------
# Experiments
# -----------------

lsr_exp = get_experiment_dict(
    [('z-1-smooth', 'y-1-smooth', 'x-1-smooth', 'mask', 'centreness-log'), 
     ('z-1', 'y-1', 'x-1', 'mask', 'centreness-log')], 
    ['z-1s_y-1s_x-1s_m_cl', 'z-1_y-1_x-1_m_cl'], 
    name='label-smoothing-reg-exp'
)

affinities_exp = get_experiment_dict(
    [('z-1', 'y-1', 'x-1', 'mask', 'centreness-log'), 
     ('z-1', 'z-2', 'y-1', 'y-2', 'x-1', 'x-2', 'mask', 'centreness-log'),
     ('z-1', 'z-3', 'z-2', 'y-1', 'y-2', 'y-3', 'x-1', 'x-2', 'x-3', 'mask', 'centreness-log')], 
    ['z-1_y-1_x-1_m_cl', 'z-1_z-2_y-1_y-2_x-1_x-2_m_cl', 'z-1_z-2_z-3_y-1_y-2_y-3_x-1_x-2_x-3_m_cl'], 
     name='affinities-exp'
)

thresh_exp = get_experiment_dict(
    [('z-1', 'y-1', 'x-1', 'mask', 'centreness-log'), 
     ('z-1', 'y-1', 'x-1', 'centreness', 'centreness-log')], 
    ['z-1_y-1_x-1_m_cl', 'z-1_y-1_x-1_c_cl'], 
    name='threshold-exp'
)

forked_exp = get_experiment_dict(
    [('z-1', 'y-1', 'x-1', 'mask', 'centreness-log'), 
     ('z-1', 'y-1', 'x-1', 'mask', 'centreness-log')], 
    ['z-1_y-1_x-1_m_cl', 'f3,2_z-1_y-1_x-1_m_cl'], 
    [{}, {'fork_channels': (3, 2)}], 
    name ='forked-exp'
)

seed_exp = get_experiment_dict(
    [('z-1', 'y-1', 'x-1', 'mask', 'centreness'), 
     ('z-1', 'y-1', 'x-1', 'mask', 'centreness-log'),
     ('z-1', 'y-1', 'x-1', 'mask', 'centroid-gauss')],
    ['z-1_y-1_x-1_m_c', 'z-1_y-1_x-1_m_cl', 'z-1_y-1_x-1_m_cg'], 
    name='seed-exp'
)

loss_exp = get_experiment_dict(
    [('z-1', 'y-1', 'x-1', 'mask', 'centreness-log'), 
     ('z-1', 'y-1', 'x-1', 'mask', 'centreness-log')],
    ['BCE_z-1_y-1_x-1_m_cl', 'DICE_z-1_y-1_x-1_m_cl'], 
    [{'loss_function' : 'BCELoss'}, {'loss_function' : 'DICELoss'}], 
    name='loss-exp'
)

lr_exp = get_experiment_dict(
    [('z-1', 'y-1', 'x-1', 'mask', 'centreness-log'), 
     ('z-1', 'y-1', 'x-1', 'mask', 'centreness-log'),
     ('z-1', 'y-1', 'x-1', 'mask', 'centreness-log')], 
    ['lr0-05_z-1_y-1_x-1_m_cl', 'lr0-01_z-1_y-1_x-1_m_cl', 'lr0-005_z-1_y-1_x-1_m_cl'], 
    [{'lr' : 0.05}, {'lr' : 0.01}, {'lr' : 0.005}], 
    name='learning-rate-exp'
)

mini_exp = get_experiment_dict(
    [('z-1', 'y-1', 'x-1', 'mask', 'centreness-log')], 
    ['z-1_y-1_x-1_m_c'], 
    [{'epochs' : 2}], 
    n_each=25, 
    name='mini-train-unet'
)

basic_exp = get_experiment_dict(
    [('z-1', 'y-1', 'x-1', 'mask', 'centreness-log')], 
    ['z-1_y-1_x-1_m_c'], 
    n_each=50
)

def get_files(dirs, ends='.zarr'):
    files = []
    for d in dirs:
        for sub in os.walk(d):
            if ends.endswith('.zarr'):
                if sub[0].endswith(ends):
                    files.append(sub[0])
            for fl in sub[2]:
                f = os.path.join(sub[0], fl)
                if f.endswith(ends):
                    files.append(f)
    return files

if __name__ == '__main__':
 
    # New code
    out_dir = '/home/abigail/data/plateseg-training/training_output'
    dirs = ['/home/abigail/data/plateseg-training/training_gt/Pia', 
            '/home/abigail/data/plateseg-training/training_gt/Volga', 
            '/home/abigail/data/plateseg-training/training_gt/Abi']
    image_paths = get_files(dirs, ends='_image.zarr')
    gt_paths = get_files(dirs, ends='_labels.zarr')
    unets = run_experiment(basic_exp, image_paths, gt_paths, out_dir)
    