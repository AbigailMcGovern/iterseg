from matplotlib.colors import same_color
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from pathlib import Path
import re
from .plots import VI_plot, plot_AP, plot_count_difference

import pandas as pd
from skimage.measure import regionprops
from skimage.metrics import variation_of_information
import umetrix

import napari
from scipy import stats

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
    gt_data: napari.layers.Labels,
    model_result:  napari.layers.Labels,
    name: str,
    prefix : str,
    VI: bool = True, 
    AP: bool = True, 
    ND: bool = True,
    out_path = None,
    exclude_chunks: int = 10, 
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
    scores = {
        'VI: GT | Output' : [], 
        'VI: Output | GT' : [], 
        'Number objects (GT)' : [], 
        'Number objects (model)' : [], 
        'Count difference' : [], 
        'Count difference (%)' : [], 
        }
    IoU_dict = generate_IoU_dict()
    scores.update(IoU_dict)
    if isinstance(gt_data, napari.layers.Labels):
        gt_data = gt_data.data
    if isinstance(model_result, napari.layers.Labels):
        model_result = model_result.data
    if gt_data.ndim != model_result.ndim:
        dim_dif = gt_data.ndim - model_result.ndim
        if dim_dif == -1:
            # if 3D and 4D will assume 3D needs to be replicated t times (data is tzyx) and stacked
            gt_data = [gt_data, ] * model_result.shape[0]
            gt_data = np.stack(gt_data)
        elif dim_dif == 1:
            model_result = [model_result, ] * gt_data.shape[0]
            model_result = np.stack(model_result)
        else:
            raise ValueError('Ground truth and model result must be either 3D or 4D arrays')
    for s_, c_ in slices:
        gt = gt_data[s_]
        gt = np.squeeze(gt)[c_]
        n_objects = np.unique(gt).size
        if n_objects > exclude_chunks + 1:
            mr = model_result[s_]
            mr = np.squeeze(mr)[c_]
            #print('n_objects', n_objects)
            if VI:
                vi = variation_of_information(gt, mr)
                scores['VI: GT | Output'].append(vi[0])
                scores['VI: Output | GT'].append(vi[1])
            if AP:
                generate_IoU_data(gt, mr, scores)
            if ND:
                n_mr = np.unique(mr).size
                #print('n_mr', n_mr, np.unique(mr), mr.shape, mr.dtype)
                nd = n_mr - n_objects
                ndp = nd / n_objects * 100 # as a percent might be more informative
                scores['Count difference (%)'].append(ndp)
                scores['Number objects (GT)'].append(n_objects)
                scores['Number objects (model)'].append(n_mr)
                scores['Count difference'].append(nd)
    lens = {key : len(scores[key]) for key in scores.keys()}
    to_keep = [key for key in scores.keys() if lens[key] > 1]
    new_scores = {key : scores[key] for key in to_keep}
    new_scores = pd.DataFrame(new_scores)
    statistics = single_sample_stats(new_scores, to_keep, name)
    new_scores['model_name'] = [name, ] * len(new_scores)
    if out_path is not None:
        n = prefix + '_' + name + '_scores.csv'
        scores_path = os.path.join(out_path, n)
        new_scores.to_csv(scores_path)
        statistics = statistics.T
        n = prefix + '_' + name + '_stats.csv'
        stats_p = os.path.join(out_path, n)
        statistics.to_csv(stats_p)
    ap_scores = None
    if AP:
        ap_scores = generate_ap_scores(new_scores, name)
        if out_path is not None:
            n = prefix + '_' + name + '_AP_curve.csv'
            AP_path = os.path.join(out_path, n)
            ap_scores.to_csv(AP_path)
    return (new_scores, ap_scores), statistics


def single_sample_stats(df, columns, name):
    results = {}
    alpha = 0.95
    for c in columns:
        sample_mean = np.mean(df[c].values)
        sample_sem = stats.sem(df[c].values)
        degrees_freedom = df[c].values.size - 1
        CI = stats.t.interval(alpha, degrees_freedom, sample_mean, sample_sem)
        n = str(c) + '_'
        results[n + 'mean'] = [sample_mean, ]
        results[n + 'sem'] = [sample_sem, ]
        results[n + '95pcntCI_2-5pcnt'] = [CI[0], ]
        results[n + '95pcntCI_97-5pcnt'] = [CI[1], ]
    results = pd.DataFrame(results)
    results['model_name'] = name
    return results


#def metrics_for_stack(directory, name, seg, gt):
#    assert seg.shape[0] == gt.shape[0]
#    IoU_dict = generate_IoU_dict()
#    for i in range(seg.shape[0]):
#        seg_i = seg[i].compute()
#        gt_i = gt[i].compute()
#        generate_IoU_data(gt_i, seg_i, IoU_dict)
#    df = save_data(IoU_dict, name, directory, 'metrics')
#    ap = generate_ap_scores(df, name, directory)
#    return df, ap


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
        result = umetrix.calculate(gt, seg, strict=True, iou_threshold=t)
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


def save_data(data_dict, name, prefix, directory, suffix):
    df = pd.DataFrame(data_dict)
    n = prefix + '_' + name + '_' + suffix +'.csv'
    p = os.path.join(directory, n)
    df.to_csv(p)
    return df


def generate_ap_scores(
        df, 
        name,  
        thresholds=(0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9)
        ):
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
    ap_scores['model_name'] = [name, ] * len(thresholds)
    ap_scores = pd.DataFrame(ap_scores)
    #print(ap_scores)
    return ap_scores



# ----------------
# Plotting Methods
# ----------------
def plot_accuracy_metrics(
    data: tuple,
    prefix: str,
    save_dir: str,
    name: str,
    variation_of_information: bool, 
    average_precision: bool, 
    object_count: bool,
    show: bool =True
    ):
    '''
    Parameters
    ----------
    data: tuple of pd.DataFrame
        df0 w/ VI/IoU/nd data, df1 w/ AP data
    prefix: str
        name under which to 
    save_dir: str
        directory into which to save plots
    show: bool
        show the plots?
    '''
    df0 = data[0]
    df1 = data[1]
    if variation_of_information:
        n = prefix + '_' + name + '_VI_plot.pdf'
        VI_path = os.path.join(save_dir, n)
        VI_plot(
                df0, 
                cond_ent_over='VI: GT | Output', 
                cond_ent_under='VI: Output | GT', 
                save=VI_path, show=show)
    if average_precision:
        n = prefix + '_' + name + '_AP_plot.pdf'
        AP_path = os.path.join(save_dir, n)
        df1 = [df1, ]
        plot_AP(df1, [prefix, ], AP_path, 'Average precision', show=show)
    if object_count:
        n = prefix + '_' + name + '_OD_plot.pdf'
        OD_path = os.path.join(save_dir, n)
        plot_count_difference(df0, 'Object count difference', OD_path, show=show)



# ---------
# OLD STUFF
# ---------

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


# --------------------------
# Experimental image metrics
# --------------------------

def affinity_sum_graph(img, affs=(1, 2, 3, 5, 10, 20, 40)):
    results = []
    dims = len(img.shape)
    for a in affs:
        sums = []
        for ax in range(dims):
            diff = np.diff(img, n=a, axis=ax)
            norm_sum = np.abs(np.sum(diff)/diff.size)
            sums.append(norm_sum)
        results.append(np.sum(sums))
    return list(affs), results
            


# --------------
# Object Metrics
# --------------


def get_object_metrics():
    pass
