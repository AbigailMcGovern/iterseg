from os import path
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
#import ptitprince as pt
import seaborn as sns
from typing import Union
import matplotlib

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
            df, 
            cond_ent_over="GT | Output",  
            cond_ent_under="Output | GT", 
            lab='Variation of information',
            save=False, 
            show=True, 
            ax=None, 
            title=True, 
            palette='Set2', 
            orient='h', 
            sigma=0.2, 
            compare=False
            ):
    #df = pd.read_csv(path)
    overseg = df[cond_ent_over].values
    o_groups = [cond_ent_over] * len(overseg)
    underseg = df[cond_ent_under].values
    u_groups = [cond_ent_under] * len(underseg)
    groups = o_groups + u_groups
    x = lab
    y = 'Conditional entropy'
    data = {
        x : groups, 
        y : np.concatenate([overseg, underseg])
        }
    data = pd.DataFrame(data)
    if ax is None:
        f, ax = plt.subplots(figsize=(8, 6))
    
    sns.boxplot(x=x, y=y, data=data, palette=palette, ax=ax)
    sns.stripplot(x=x, y=y, data=data, palette=palette, edgecolor="white", ax=ax,
                 size=3, jitter=1, zorder=0, orient=orient,  dodge=True, linewidth=0.3)
    sns.boxplot()
    #pt.RainCloud(x=x, y=y, data=data, palette=palette, bw=sigma,
    #             width_viol=.6, ax=ax, orient=orient)
    p = Path(save)
    if title:
        plt.title(p.stem)
    if save:
        plt.savefig(save, bbox_inches='tight')
    if show:
        plt.show()



def VI_plot_compare(
            df, 
            ax0,
            ax1, 
            comparison_name,
            conditions,
            cond_ent_over="VI: GT | Output",  
            cond_ent_under="VI: Output | GT", 
            palette='Set2', 
            orient='h', 
            sigma=0.2,
            name='model_name',
            ):
    sns.boxplot(x=name, y=cond_ent_over, data=df, palette=palette, ax=ax0)
    sns.stripplot(x=name, y=cond_ent_over, data=df, palette=palette, edgecolor="white", ax=ax0,
                 size=3, jitter=1, zorder=0, orient=orient,  dodge=True, linewidth=0.3)
    #pt.RainCloud(x=name, y=cond_ent_over, hue=name, hue_order=conditions, data=df, palette=palette, bw=sigma,
     #            width_viol=.6, ax=ax0, orient=orient, alpha=0.8, move=0.3)
    ax0.set_ylabel(comparison_name)
    sns.despine(ax=ax0)
    ax0.legend([],[], frameon=False)
    sns.boxplot(x=name, y=cond_ent_under, data=df, palette=palette, ax=ax1)
    sns.stripplot(x=name, y=cond_ent_under, data=df, palette=palette, edgecolor="white", ax=ax1,
                 size=3, jitter=1, zorder=0, orient=orient,  dodge=True, linewidth=0.3)
    #pt.RainCloud(x=name, y=cond_ent_under, hue=name, hue_order=conditions, data=df, palette=palette, bw=sigma,
    #             width_viol=.6, ax=ax1, orient=orient, alpha=0.8, move=0.3)
    ax1.set_ylabel(comparison_name)
    sns.despine(ax=ax1)
    ax1.legend([],[], frameon=False)



def experiment_VI_plots(
        dfs, 
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
    for i, df in enumerate(dfs):
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
    f, axs = plt.subplots(1, 2, figsize=(8, 6)) #, sharex=True) #, sharey=True)
    ax0 = axs[0]
    ax1 = axs[1]
    sns.boxplot(x=x, y=cond_ent_over, data=data, palette=pal, ax=ax0)
    sns.stripplot(x=x, y=cond_ent_over, data=data, palette=pal, edgecolor="white", ax=ax0,
                 size=3, jitter=1, zorder=0, orient=o, dodge=True, linewidth=0.3)
    #pt.RainCloud(x = x, y = cond_ent_over, data = data, palette = pal, bw = sigma,
    #             width_viol = .6, ax = ax0, orient = o, move=0.3)
    ax0.set_title('Over-segmentation conditional entropy')
    sns.boxplot(x=x, y=cond_ent_under, data=data, palette=pal, ax=ax0)
    sns.stripplot(x=x, y=cond_ent_under, data=data, palette=pal, edgecolor="white", ax=ax0,
                 size=3, jitter=1, zorder=0, orient=o, dodge=True, linewidth=0.3)
    #pt.RainCloud(x = x, y = cond_ent_under, data = data, palette = pal, bw = sigma,
    #             width_viol = .6, ax = ax1, orient = o, move=0.3)
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



def plot_AP(dfs, names, out_path, title, 
            thresh_name='threshold', ap_name='average_precision', 
            show=True, add_title=True, ):
    plt.rcParams.update({'font.size': 16})
    plt.rcParams["figure.figsize"] = (8,8)
    fig = plt.figure()
    for df in dfs:
        plt.plot(df[thresh_name].values, df[ap_name].values)
    plt.xlabel('IoU threshold')
    plt.ylabel('Average precision')
    if add_title:
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
    sns.boxplot(x=x, y='n_diff', data=data, palette=pal, ax=ax)
    sns.stripplot(x=x, y='n_diff', data=data, palette=pal, edgecolor="white", ax=ax,
                 size=3, jitter=1, zorder=0, orient=o, dodge=True, linewidth=0.3)
    #pt.RainCloud(x=x, y='n_diff', data=data, palette=pal, bw=sigma,
    #             width_viol=.6, ax=ax, orient=o)
    plt.title(title)
    f.savefig(out_path)
    if show:
        plt.show()
    

def plot_count_difference(
        df, 
        title, 
        out_path, 
        col_name='Count difference', 
        show=True, 
        ):
    plt.rcParams.update({'font.size': 16})
    groups = ['model', ] * len(df)
    n_diff = df[col_name].values
    #for i, df in enumerate(dfs):
     #   vals = df[col_name].values
      #  n_diff.append(vals)
       # groups += [names[i]] * len(df)
    x = 'Experiment'
    data = {
        x : groups, 
        'n_diff' : n_diff, 
    }
    data = pd.DataFrame(data)
    o = 'h'
    pal = 'Set2'
    sigma = .2
    f, ax = plt.subplots(figsize=(10, 10))
    sns.boxplot(x=x, y='n_diff', data=data, palette=pal, ax=ax)
    sns.stripplot(x=x, y='n_diff', data=data, palette=pal, edgecolor="white", ax=ax,
                 size=3, jitter=1, zorder=0, orient=o, dodge=True, linewidth=0.3)
    #pt.RainCloud(x=x, y='n_diff', data=data, palette=pal, bw=sigma,
     #            width_viol=.6, ax=ax, orient=o)
    plt.title(title)
    f.savefig(out_path)
    if show:
        plt.show()



def compare_count_difference(
        df, 
        ax,
        comparison_name,
        conditions, 
        col_name='Count difference', 
        palette='Set2', 
        orient='h', 
        sigma=0.2,
        name='model_name',
        ):
    conditions = pd.unique(df[name])
    sns.boxplot(x=name, y=col_name, data=df, palette=palette, ax=ax)
    sns.stripplot(x=name, y=col_name, data=df, palette=palette, edgecolor="white", ax=ax,
                 size=3, jitter=1, zorder=0, orient=orient, dodge=True, linewidth=0.3)
    #pt.RainCloud(x=name, y=col_name, hue=name, hue_order=conditions, data=df, palette=palette, bw=sigma,
    #             width_viol=.6, ax=ax, orient=orient, alpha=0.8, move=0.3)
    ax.set_ylabel(comparison_name)
    sns.despine(ax=ax)
    ax.legend([],[], frameon=False)
    


def compare_AP(
        df, 
        ax, 
        palette, 
        conditions,
        name='model_name', 
        ap_col='average_precision', 
        thresh_col='threshold'
        ):
    conditions = pd.unique(df[name])
    sns.lineplot(x=thresh_col, y=ap_col, hue=name, hue_order=conditions, data=df, ax=ax, palette=palette)
    ax.set_xlabel('IOU threshold')
    ax.set_ylabel('Average precision')
    sns.despine(ax=ax)


def comparison_plots(
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
        raincloud_orientation: str='h',
        raincloud_sigma: float=0.2,
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
    # Read in tables and collate data
    # -------------------------------
    VIOD_files = [os.path.join(comparison_directory, f) \
                  for f in os.listdir(comparison_directory) \
                    if f.endswith('_scores.csv')]
    metrics_VIOD = [pd.read_csv(p) for p in VIOD_files]
    metrics_VIOD = pd.concat(metrics_VIOD).reset_index(drop=True)
    AP_files = [os.path.join(comparison_directory, f) \
                  for f in os.listdir(comparison_directory) \
                    if f.endswith('_AP_curve.csv')]
    metrics_AP = [pd.read_csv(p) for p in AP_files]
    metrics_AP = pd.concat(metrics_AP).reset_index(drop=True)
    
    conditions = pd.unique(metrics_VIOD['model_name'])
    
    # plotting
    # --------
    matplotlib.rcParams.update({'font.size': font_size})
    sns.set_context(context)
    sns.set_style(style)
    plt.rcParams['svg.fonttype'] = 'none' # so the text will be saved with the svg - not curves
    is_int = []
    if variation_of_information:
        is_int.append(isinstance(VI_indexs[0], int))
        is_int.append(isinstance(VI_indexs[1], int))
    if average_precision:
        is_int.append(isinstance(AP_index, int))
    if object_difference:
        is_int.append(isinstance(OD_index, int))
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_col)
    if np.sum(is_int) == len(is_int):
        axs = axs.ravel()
    fig.set_size_inches(fig_size)
    #plt.rcParams.update({'font.size': font_size})
    if variation_of_information:
        VI_plot_compare(metrics_VIOD, axs[VI_indexs[0]], axs[VI_indexs[1]], 
                        comparison_name, conditions,
                        palette=palette, 
                        orient=raincloud_orientation, 
                        sigma=raincloud_sigma, 
                        name='model_name')
    if object_difference:
        compare_count_difference(metrics_VIOD, axs[OD_index], comparison_name, conditions,
                                 col_name='Count difference', 
                                 palette=palette, orient=raincloud_orientation, 
                                 sigma=raincloud_sigma, name='model_name')
    if average_precision:
        compare_AP(metrics_AP, axs[AP_index], palette, conditions, name='model_name', 
                   ap_col='average_precision', thresh_col='threshold')

    # White space
    # -----------
    right = 1 - (right_white_space / 100)
    top = 1 - (top_white_space / 100)
    left = left_white_space / 100
    bottom = bottom_white_space / 100
    wspace = horizontal_white_space / 100
    hspace = vertical_white_space / 100
    fig.subplots_adjust(right=right, left=left, bottom=bottom, top=top, wspace=wspace, hspace=hspace)
    
    matplotlib.rcParams.update({'font.size': font_size})
    
    # Save figure
    # -----------
    if output_directory is None:
        output_directory = comparison_directory
    n = save_name + '.' + file_exstention
    save_path = os.path.join(output_directory, n)
    fig.savefig(save_path)

    if show:
        plt.show()