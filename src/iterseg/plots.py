from os import path
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
import ptitprince as pt

# ----------
# Loss Plots
# ----------

def save_loss_plot(path, loss_function, v_path=None, show=True):
    df = pd.read_csv(path)
    if v_path is not None:
        vdf = pd.read_csv(v_path)
    else:
        vdf = None
    p = Path(path)
    n = p.stem
    d = p.parents[0]
    out_path = os.path.join(d, n + '_loss.png')
    fig, ax = plot_loss(df, vdf=vdf, x_lab='Iteration', y_lab=loss_function, save=out_path, show=show)



def plot_loss(df, vdf=None, x_lab='Iteration', y_lab='BCE Loss', save=None, show=True):
    x = df['Unnamed: 0'].values
    y = df['loss'].values
    epochs = len(df['epoch'].unique())
    no_batches = int(len(x) / epochs)
    epoch_ends = np.array([((i + 1) * no_batches) - 1 for i in range(epochs)])
    epoch_end_x = x[epoch_ends]
    epoch_end_y = y[epoch_ends]
    fig, ax = plt.subplots()
    leg = ['loss',]
    ax.plot(x, y, linewidth=2)
    ax.scatter(epoch_end_x, epoch_end_y)
    title = 'Training loss'
    if vdf is not None:
        if len(vdf) > epochs:
            vy = vdf.groupby('batch_id').mean()['validation_loss'].values
            vx = vdf['batch_id'].unique()
        else:
            vy = vdf['validation_loss'].values
            vx = epoch_end_x
        title = title + ' with validation loss'
        leg.append('validation loss')
        if len(vdf) > epochs:
            #vy_err = v_df.groupby('batch_id').sem()['validation_loss'].values
            #ax.errorbar(vx, vy, vy_err, marker='.')
            ax.plot(vx, vy, linewidth=2, marker='o')
        else:
            ax.plot(vx, vy, linewidth=2, marker='o')
    ax.set(xlabel=x_lab, ylabel=y_lab)
    ax.set_title(title)
    ax.legend(leg)
    fig.set_size_inches(13, 9)
    if save is not None:
        plt.savefig(save, dpi=300)
    if show:
        plt.show()
    return fig, ax



def save_channel_loss_plot(path, show=True):
    df = pd.read_csv(path)
    p = Path(path)
    n = p.stem
    d = p.parents[0]
    out_path = os.path.join(d, n + '_channel-loss.png')
    fig, ax = plot_channel_losses(df, save=out_path, show=show)



def plot_channel_losses(df, x_lab='Iteration', y_lab='BCE Loss', save=None, show=True):
    cols = list(df.columns)
    x = df['Unnamed: 0'].values
    non_channel_cols = ['Unnamed: 0', 'epoch', 'batch_num', 'loss', 'data_id']
    channel_losses = [col for col in cols if col not in non_channel_cols]
    #print(channel_losses)
    if len(channel_losses) > 5:
        #print('four plots')
        fig, axs = plt.subplots(2, 2)
        zs, ys, xs, cs = [], [], [], []
        for col in channel_losses:
            y = df[col].values
            if col.startswith('z'):
                ls = _get_linestyle(zs)
                axs[0, 0].plot(x, y, linewidth=1, linestyle=ls)
                zs.append(col)
            if col.startswith('y'):
                ls = _get_linestyle(ys)
                axs[0, 1].plot(x, y, linewidth=1, linestyle=ls)
                ys.append(col)
            if col.startswith('x'):
                ls = _get_linestyle(xs)
                axs[1, 0].plot(x, y, linewidth=1, linestyle=ls)
                xs.append(col)
            if col.startswith('cent') or col == 'mask':
                ls = _get_linestyle(cs)
                axs[1, 1].plot(x, y, linewidth=1, linestyle=ls)
                cs.append(col)
        axs[0, 0].set_title('Z affinities losses')
        axs[0, 0].legend(zs)
        axs[0, 1].set_title('Y affinities losses')
        axs[0, 1].legend(ys)
        axs[1, 0].set_title('X affinities losses')
        axs[1, 0].legend(xs)
        axs[1, 1].set_title('Object interior losses')
        axs[1, 1].legend(cs)
        fig.set_size_inches(13, 9)
    elif len(channel_losses) <= 5:
        #print('two plots')
        fig, axs = plt.subplots(2, 1)
        affs, cs = [], []
        for col in channel_losses:
            y = df[col].values
            if col.startswith('z') or col.startswith('y') or col.startswith('x'):
                ls = _get_linestyle(affs)
                axs[0].plot(x, y, linewidth=2, linestyle=ls)
                affs.append(col)
            if col.startswith('cent') or col == 'mask':
                axs[1].plot(x, y, linewidth=2)
                cs.append(col)
        axs[0].set_title('Affinities losses')
        axs[0].legend(affs)
        axs[1].set_title('Object interior losses')
        axs[1].legend(cs)
        fig.set_size_inches(14, 14)
    for ax in axs.flat:
        ax.set(xlabel=x_lab, ylabel=y_lab)
    if save is not None:
        plt.savefig(save, dpi=300)
    if show:
        plt.show()
    return fig, axs


def _get_linestyle(lis):
    if len(lis) == 0:
        ls = '-'
    elif len(lis) == 1:
        ls = '--'
    else:
        ls = ':'
    return ls


# --------
# VI Plots
# --------


def VI_plot(
            path, 
            cond_ent_over="GT | Output",  
            cond_ent_under="Output | GT", 
            lab="",
            save=False, 
            show=True):
    df = pd.read_csv(path)
    overseg = df[cond_ent_over].values
    o_groups = [cond_ent_over] * len(overseg)
    underseg = df[cond_ent_under].values
    u_groups = [cond_ent_under] * len(underseg)
    groups = o_groups + u_groups
    x = 'Variation of information'
    y = 'Conditional entropy'
    data = {
        x : groups, 
        y : np.concatenate([overseg, underseg])
        }
    data = pd.DataFrame(data)
    o = 'h'
    pal = 'Set2'
    sigma = .2
    f, ax = plt.subplots(figsize=(12, 10))
    pt.RainCloud(x = x, y = y, data = data, palette = pal, bw = sigma,
                 width_viol = .6, ax = ax, orient = o)
    p = Path(path)
    plt.title(p.stem)
    if save:
        save_path = os.path.join(p.parents[0], p.stem + lab + '_VI_rainclout_plot.png')
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()


def experiment_VI_plots(
        paths, 
        names, 
        title,
        out_name,
        out_dir,
        cond_ent_over="GT | Output",  
        cond_ent_under="Output | GT", 
        show=True
    ):
    plt.rcParams.update({'font.size': 16})
    groups = []
    ce0 = []
    ce1 = []
    for i, p in enumerate(paths):
        df = pd.read_csv(p)
        ce0.append(df[cond_ent_over].values)
        ce1.append(df[cond_ent_under].values)
        groups += [names[i]] * len(df)
    x = 'Experiment'
    data = {
        x : groups, 
        cond_ent_over : np.concatenate(ce0), 
        cond_ent_under : np.concatenate(ce1)
    }
    data = pd.DataFrame(data)
    o = 'h'
    pal = 'Set2'
    sigma = .2
    f, axs = plt.subplots(1, 2, figsize=(14, 10)) #, sharex=True) #, sharey=True)
    ax0 = axs[0]
    ax1 = axs[1]
    pt.RainCloud(x = x, y = cond_ent_over, data = data, palette = pal, bw = sigma,
                 width_viol = .6, ax = ax0, orient = o)
    ax0.set_title('Over-segmentation conditional entropy')
    pt.RainCloud(x = x, y = cond_ent_under, data = data, palette = pal, bw = sigma,
                 width_viol = .6, ax = ax1, orient = o)
    ax1.set_title('Under-segmentation conditional entropy')
    f.suptitle(title)
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, out_name + '_VI_rainclould_plots.png')
    plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()



# -----------------------
# Average Precision Plots
# -----------------------

def plot_experiment_APs(paths, names, title, out_dir, out_name, show=True):
    dfs = [pd.read_csv(path) for path in paths]
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_name)
    plot_AP(dfs, names, out_path, title, show=show)


def plot_AP(dfs, names, out_path, title, thresh_name='threshold', ap_name='average_precision', show=True):
    plt.rcParams.update({'font.size': 16})
    plt.rcParams["figure.figsize"] = (10,10)
    fig = plt.figure()
    for df in dfs:
        plt.plot(df[thresh_name].values, df[ap_name].values)
    plt.xlabel('IoU threshold')
    plt.ylabel('Average precision')
    plt.title(title)
    plt.legend(names)
    fig.savefig(out_path)
    if show:
        plt.show()


# ------------------------------
# Object Number Difference Plots
# ------------------------------

def plot_experiment_no_diff(paths, names, title, out_dir, out_name, col_name='n_diff', show=True):
    dfs = [pd.read_csv(path) for path in paths]
    plt.rcParams.update({'font.size': 16})
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_name)
    groups = []
    n_diff = []
    for i, df in enumerate(dfs):
        vals = df[col_name].values
        n_diff.append(vals)
        groups += [names[i]] * len(df)
    x = 'Experiment'
    data = {
        x : groups, 
        'n_diff' : np.concatenate(n_diff), 
    }
    data = pd.DataFrame(data)
    o = 'h'
    pal = 'Set2'
    sigma = .2
    f, ax = plt.subplots(figsize=(10, 10))
    pt.RainCloud(x=x, y='n_diff', data=data, palette=pal, bw=sigma,
                 width_viol=.6, ax=ax, orient=o)
    plt.title(title)
    f.savefig(out_path)
    if show:
        plt.show()
    


if __name__ == '__main__':
    import re
    #name = 'loss_z-1_z-2_y-1_y-2_y-3_x-1_x-2_x-3_c_cl.csv'
    #name = 'loss_210401_150158_z-1_y-1_x-1__wBCE2-1-1.csv'
    #dir_ = '/Users/amcg0011/Data/pia-tracking/cang_training/210331_training_0'
    #dir_ = '/Users/amcg0011/Data/pia-tracking/cang_training/210401_150158_z-1_y-1_x-1__wBCE2-1-1'
    root_dir = '/home/abigail/data/platelet-segmentation-training'
    pattern = re.compile(r'\d{6}_\d{6}_')
    loss_pattern = re.compile(r'loss_\d{6}_\d{6}_')
    val_loss_pattern = re.compile(r'validation-loss_\d{6}_\d{6}_')
    date = '210513'
    for d in os.listdir(root_dir):
        mo = pattern.search(d)
        if mo is not None and d.startswith(date):
            train_dir = os.path.join(root_dir, d)
            for f in os.listdir(train_dir):
                #print(f)
                l_mo = loss_pattern.search(f)
                vl_mo = val_loss_pattern.search(f)
                if vl_mo is not None and f.endswith('.csv'):
                    val_loss_path = os.path.join(train_dir, f)
                elif l_mo is not None and f.endswith('.csv'):
                    loss_path = os.path.join(train_dir, f)
                if f == 'log.txt':
                    with open(os.path.join(train_dir, f)) as log:
                        lines = log.readlines()
                        for line in lines:
                            if line.startswith('Loss function:'):
                                loss_function = line[15:-1]
            # check that we found the loss files
            do = True
            try:
                assert loss_path is not None
            except:
                print('Could not find losses for ', d)
                do = False
            try:
                assert val_loss_path is not None
            except:
                print('Could not find validation losses for ', d)
                do = False
            if loss_function is None:
                loss_function = 'Unknown'
                print('could not find loss function in training log')
            # save the loss plots
            if do:
                print(d)
                save_channel_loss_plot(loss_path)
                print('Saved channel losses for ', d)
                save_loss_plot(loss_path, loss_function, val_loss_path)
                print('Saved loss plots for ', d)

    #path = os.path.join(dir_, name)
    #save_channel_loss_plot(path)
    #v_name = 'validation-loss_z-1_z-2_y-1_y-2_y-3_x-1_x-2_x-3_c_cl.csv'
    #v_name = 'validation-loss_210401_150158_z-1_y-1_x-1__wBCE2-1-1.csv'
    #v_path = os.path.join(dir_, v_name)
    #loss_function = 'Weighted BCE Loss (2, 1, 1)'
    #save_loss_plot(path, loss_function, v_path)
