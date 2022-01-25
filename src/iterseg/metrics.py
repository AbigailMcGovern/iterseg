import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from os import path
import re
from .plots import experiment_VI_plots, VI_plot, plot_experiment_APs, plot_experiment_no_diff

import pandas as pd
from skimage.measure import regionprops
from skimage.metrics import variation_of_information
import umetrics

from napari.types import LabelsData

# compare model result with ground truth validation data (not seen by network)

# SEG METRICS
#   - Average precision
#   - Variation of information
#   - Object count difference


# plot accuracy metrics against the following to analyse where models fail
#   because #visualanalyticsbro

# IM METRICS
#   - image entropy
#   - average intensity

# OB METICS
#   - average volume
#   - density
#   - average elongation/flattness



# --------------------
# Segmentation Metrics
# --------------------


def get_accuracy_metrics(
    slices, 
    gt_data: LabelsData,
    model_result: LabelsData,
    VI: bool = True, 
    AP: bool = True, 
    ND: bool = True,
    out_path = None,
    ):
    '''
    Parameters:
    slices: list of tupel of 4 x slice 
        Slices denoting where the image chunks to be taken as single units of
        network output end and begin.
    gt_data: napari.types.LabelsData or int array like
        Stack of N x 3D validation ground truth (i.e., 4D with N,z,y,x)
    model_result: napari.types.LabelsData or int array like
        Stack of N x 3D model derived segmentation (i.e., 4D with N,z,y,x)
    VI: bool
        Should we find variation of information scores
    AP: bool
        Should we find average precision scores (for IoU 0.6-0.9)
    ND: bool
        SHould we find the number of objects difference from 
    '''
    scores = {'VI: GT | Output' : [], 'VI: Output | GT' : [], 'Number difference' : []}
    IoU_dict = generate_IoU_dict()
    scores.update(IoU_dict)
    for s_ in slices:
        gt = gt_data[s_]
        mr = model_result[s_]
        if VI:
            vi = variation_of_information(gt, mr)
            scores['GT | Output'].append(vi[0])
            scores['Output | GT'].append(vi[1])
        if AP:
            generate_IoU_data(gt, mr, scores)
        if ND:
            n_gt = np.unique(gt).size
            n_mr = np.unique(mr).size
            nd = n_gt - n_mr
            scores['Number difference'].append(nd)
    to_keep = [key for key in scores.keys() if len(scores[key]) == len(slices)]
    new_scores = {key : scores[key] for key in to_keep}
    new_scores = pd.DataFrame(new_scores)
    if out_path is not None:
        new_scores.to_csv(out_path)
    return new_scores


def metrics_for_stack(directory, name, seg, gt):
    assert seg.shape[0] == gt.shape[0]
    IoU_dict = generate_IoU_dict()
    for i in range(seg.shape[0]):
        seg_i = seg[i].compute()
        gt_i = gt[i].compute()
        generate_IoU_data(gt_i, seg_i, IoU_dict)
    df = save_data(IoU_dict, name, directory, 'metrics')
    ap = generate_ap_scores(df, name, directory)
    return df, ap


def calc_ap(result):
        denominator = result.n_true_positives + result.n_false_negatives + result.n_false_positives
        return result.n_true_positives / denominator


def generate_IoU_dict(thresholds=(0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9)):
    IoU_dict = {}
    IoU_dict['n_predicted'] = []
    IoU_dict['n_true'] = []
    IoU_dict['n_diff'] = []
    for t in thresholds:
        n = f't{t}_true_positives'
        IoU_dict[n] = []
        n = f't{t}_false_positives'
        IoU_dict[n] = []
        n = f't{t}_false_negatives'
        IoU_dict[n] = []
        n = f't{t}_IoU'
        IoU_dict[n] = []
        n = f't{t}_Jaccard'
        IoU_dict[n] = []
        n = f't{t}_pixel_identity'
        IoU_dict[n] = []
        n = f't{t}_localization_error'
        IoU_dict[n] = []
        n = f't{t}_per_image_average_precision'
        IoU_dict[n] = []
    return IoU_dict


def generate_IoU_data(gt, seg, IoU_dict, thresholds=(0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9)):
    for t in thresholds:
        result = umetrics.calculate(gt, seg, strict=True, iou_threshold=t)
        n = f't{t}_true_positives'
        IoU_dict[n].append(result.n_true_positives) 
        n = f't{t}_false_positives'
        IoU_dict[n].append(result.n_false_positives) 
        n = f't{t}_false_negatives'
        IoU_dict[n].append(result.n_false_negatives) 
        n = f't{t}_IoU'
        IoU_dict[n].append(result.results.IoU) 
        n = f't{t}_Jaccard'
        IoU_dict[n].append(result.results.Jaccard) 
        n = f't{t}_pixel_identity'
        IoU_dict[n].append(result.results.pixel_identity) 
        n = f't{t}_localization_error'
        IoU_dict[n].append(result.results.localization_error) 
        n = f't{t}_per_image_average_precision'
        IoU_dict[n].append(calc_ap(result))
        if t == thresholds[0]:
            IoU_dict['n_predicted'].append(result.n_pred_labels)
            IoU_dict['n_true'].append(result.n_true_labels)
            IoU_dict['n_diff'].append(result.n_true_labels - result.n_pred_labels)


def save_data(data_dict, name, directory, suffix):
    df = pd.DataFrame(data_dict)
    n = name + '_' + suffix +'.csv'
    p = os.path.join(directory, n)
    df.to_csv(p)
    return df


def generate_ap_scores(df, name, directory, suffix, thresholds=(0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9)):
    ap_scores = {'average_precision' : [], 
                 'threshold': []}
    for t in thresholds:
        ap_scores['threshold'].append(t)
        n = f't{t}_true_positives'
        true_positives = df[n].sum()
        n = f't{t}_false_positives'
        false_positives = df[n].sum()
        n = f't{t}_false_negatives'
        false_negatives = df[n].sum()
        ap = true_positives / (true_positives + false_negatives + false_positives)
        ap_scores['average_precision'].append(ap)
    print(ap_scores)
    ap_scores = save_data(ap_scores, name, directory, suffix)
    return ap_scores



# ----------------
# Plotting Methods
# ----------------
def plot_accuracy_metrics(
    df: pd.DataFrame,
    prefix: str,
    save_dir: str,
    vi: bool, 
    ap: bool, 
    nd: bool,
    ):
    pass







# ---------
# OLD STUFF
# ---------

def segmentation_VI_plots(data_dir, seg_info, exp_name, out_name):
    vi_train_paths, vi_train_names = _get_VI_paths(data_dir, 
                                                   seg_info, 
                                                   validation=False)
    vi_val_paths, vi_val_names = _get_VI_paths(data_dir, 
                                               seg_info, 
                                               validation=True)
    for p in vi_train_paths:
        VI_plot(p, lab='_train')
    for p in vi_val_paths:
        VI_plot(p, lab='_val')
    out_dir = os.path.join(data_dir, out_name + '_VI_plots')
    experiment_VI_plots(vi_train_paths, 
                        vi_train_names, 
                        f'Training output VI Scores: {exp_name}', 
                        out_name + '_train', 
                        out_dir)
    experiment_VI_plots(vi_val_paths, 
                        vi_val_names, 
                        f'Test output VI Scores: {exp_name}', 
                        out_name + '_val', 
                        out_dir)


def segmentation_plots(data_dir, seg_info, exp_name, out_name):
    vi_paths, vi_names = _get_VI_paths(data_dir, 
                                               seg_info, 
                                               validation=True)
    ap_paths, ap_names = _get_AP_paths(data_dir, seg_info)
    nd_paths, nd_names = _get_IoU_paths(data_dir, seg_info)
    out_dir = os.path.join(data_dir, out_name)
    experiment_VI_plots(vi_paths, 
                        vi_names, 
                        f'Test output VI Scores: {exp_name}', 
                        out_name + '_val_VI', 
                        out_dir)
    plot_experiment_APs(ap_paths, 
                        ap_names, 
                        f'Average precision: {exp_name}', 
                        out_dir, 
                        out_name + '_val_AP')
    plot_experiment_no_diff(nd_paths, 
                            nd_names, 
                            f'Number difference: {exp_name}', 
                            out_dir, 
                            out_name + '_val_num-diff')



def get_experiment_seg_info(
        data_dir, 
        experiments, 
        w_scale=None, 
        compactness=0.,
        display=True,
        centroid_opt=('centreness-log', 'centreness', 'centroid-gauss'), 
        thresh_opt=('mask', 'centreness', 'centreness-log'),
        z_aff_opt=('z-1', 'z-1-smooth'),
        y_aff_opt=('y-1', 'y-1-smooth'), 
        x_aff_opt=('x-1', 'x-1-smooth'), 
        date='recent'
    ):
    seg_info = {}
    for key in experiments.keys():
        n = experiments[key]['name']
        regex = re.compile( r'\d{6}_\d{6}_' + n)
        files = os.listdir(data_dir)
        matches = []
        for f in files:
            mo = regex.search(f)
            if mo is not None:
                matches.append(mo[0])
        if date == 'recent':
            seg_dir = sorted(matches)[-1]
        else: # specific date range ??
            pass
        exp_dir = os.path.join(data_dir, seg_dir)
        if os.path.exists(exp_dir):
            seg_info[key] = {}
            # find the prefered channels for segmenting
            chans = experiments[key]['channels']
            cent_chan = _get_index(chans, centroid_opt)
            seg_info[key]['centroids_channel'] = cent_chan
            thresh_chan = _get_index(chans, thresh_opt)
            seg_info[key]['thresholding_channel'] = thresh_chan
            z_chan = _get_index(chans, z_aff_opt)
            y_chan = _get_index(chans, y_aff_opt)
            x_chan = _get_index(chans, x_aff_opt)
            seg_info[key]['affinities_channels'] = (z_chan, y_chan, x_chan)
            # FIX THIS!!! Need to use name and add a date/date range option
            # raise ValueError(f'path {exp_dir} does not exist')
            seg_info[key]['directory'] = exp_dir
            seg_info[key]['suffix'] = experiments[key]['name']
            seg_info[key]['scale'] = experiments[key]['scale']
            seg_info[key]['w_scale'] = w_scale
            seg_info[key]['compactness'] = compactness
            seg_info[key]['display'] = display
        else:
            print(f'path {exp_dir} does not exist')
    return seg_info
    

def _get_index(chans, opts):
    idx = []
    for opt in opts:
        for i, c in enumerate(chans):
            if c == opt:
                idx.append(i)
    if len(idx) == 0:
        raise ValueError(f'No channel in {chans} matches the options {opts}')
    return idx[0]


def _get_VI_paths(data_dir, seg_info, validation=False):
    if validation:
        s = 'validation_VI.csv'
    else:
        s = '_VI.csv'
    paths, names = _get_data_paths(data_dir, seg_info, s)
    return paths, names


def _get_AP_paths(data_dir, seg_info, validation=True):
    if validation:
        s = '_validation_AP.csv'
    else:
        s = '_test_AP.csv'
    paths, names = _get_data_paths(data_dir, seg_info, s)
    return paths, names
        

def _get_IoU_paths(data_dir, seg_info, validation=True):
    if validation:
        s = '_validation_metrics.csv'
    else:
        s = '_test_metrics.csv'
    paths, names = _get_data_paths(data_dir, seg_info, s)
    return paths, names


def _get_data_paths(data_dir, seg_info, s):
    paths = []
    names = []
    for key in seg_info.keys():
        dir_ = seg_info[key]['directory']
        suffix = seg_info[key]['suffix']
        p = os.path.join(dir_, suffix + s)
        if not os.path.exists(p):
            raise ValueError(f'Cannot find path {p}')
        paths.append(p)
        n = seg_info[key]['suffix']
        names.append(n)
    return paths, names



# -------------
# Image Metrics
# -------------


def get_image_metrics():
    pass



# --------------
# Object Metrics
# --------------


def get_object_metrics():
    pass
