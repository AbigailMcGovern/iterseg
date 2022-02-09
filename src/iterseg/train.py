from .custom_loss import WeightedBCELoss, DiceLoss, EpochwiseWeightedBCELoss, \
    channel_losses_to_dict, ChannelwiseLoss
from datetime import datetime
import dask.array as da
from .helpers import LINE, write_log
import napari
import numpy as np
import os
import pandas as pd
from .plots import save_loss_plot, save_channel_loss_plot
from tifffile import TiffWriter 
import torch 
import torch.nn as nn
import torch.optim as optim
from .train_io import get_train_data, load_train_data, load_dask_as_tensor, load_tensor_from_zarr
from tqdm import tqdm
from .unet import UNet, ForkedUNet


# DICE seems to be more common but BCE Loss is also used for 
#   image segmentation (Jadon et al., 2020, arXiv) and seems 
#   to give better results here


def train_unet(
               # training data
               x,
               vx,  
               y, 
               vy,
               ids=None,
               vids=None, 
               # output information
               out_dir=None, 
               name='my-unet', 
               channels=None,
               # training variables
               validate=True,
               log=True,
               epochs=3, 
               lr=0.01, 
               loss_function='BCELoss', 
               chan_weights=None, # for weighted BCE
               weights=None,
               update_every=20, 
               losses=None, 
               chan_losses=None,
               # network architechture
               fork_channels=None,
               chan_final_activations=None,
               **kwargs
               ):
    '''
    Train a basic U-Net on affinities data.

    Parameters
    ----------
    xs: list of torch.tensor
        Input images for which the network will be trained to predict
        inputted labels.
    ys: list of torch.tensor
        Input labels that represent target output that the network will
        be trained to predict.
    ids: list of str
        ID strings corresponding to xs and ys (as they are named on disk). 
        Used for saving output.
    out_dir: str
        Directory to which to save network output
    suffix: str
        Suffix used in naming pytorch state dictionary file
    channels: tuple of str or None
        Names of output channels to be used for labeling channelwise
        loss columns in output loss csv. If none, names are generated.
    v_xs: list of torch.Tensor or None
        Validation images
    v_y: list of torch.Tensor or None
        Validation labels
    v_ids: list of str or None
        Validation IDs
    validate: bool
        Will a validation be done at the end of every epoch?
    log: bool
        Will a log.txt file containing all console print outs be saved?
    epochs: int
        How many times should we go through the training data?
    lr: float
        Learning rate for Adam optimiser
    loss_function: str
        Which loss function will be used for training & validation?
        Current options include:
            'BCELoss': Binary cross entropy loss
            'WeightedBCE': Binary cross entropy loss whereby channels are weighted
                according to chan_weights parameter. Quick way to force network to
                favour learning information about a given channel/s.
            'DiceLoss': 1 - DICE coefficient of the output-target pair
    chan_weights: tuple of float
        WEIGHTEDBCE: Weights for BCE loss for each output channel. 
    weights: None or nn.Model().state_dict()
        Prior weights with which to initalise the network.
    update_every: int
        Determines how many batches are processed before printing loss

    Returns
    -------
    unet: UNet (unet.py)

    Notes
    -----
    When data is loaded from a directory, it will be recognised according
    to the following naming convention:

        IDs: YYMMDD_HHMMSS_{digit/s} 
        Images: YYMMDD_HHMMSS_{digit/s}_image.tif
        Affinities: YYMMDD_HHMMSS_{digit/s}_labels.tif
    
    E.g., 210309_152717_7_image.tif, 210309_152717_7_labels.tif

    For each ID, a labels and an image file must be found or else an
    assertion error will be raised.
    '''
    # Save?
    save_output = out_dir is not None
    print('Output will be saved: ', save_output)
    print('Save directory: ', out_dir)
    # Unique identifiers
    if ids is None:
        ids = [name + f'_{i}' for i in range(len(x))]
    if vids is None:
        vids = [name + f'_val_{i}' for i in range(len(vx))]
    # Device
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    # initialise U-net
    if fork_channels is None:
        unet = UNet(out_channels=len(channels), 
                    chan_final_activations=chan_final_activations).to(device, 
                                                                      dtype=torch.float32)
    else:
        unet = UNet(out_channels=fork_channels, 
                    chan_final_activations=chan_final_activations).to(device, 
                                                                      dtype=torch.float32)
    # load weights if applicable 
    weights_are = _load_weights(weights, unet)
    # define the optimiser
    optimiser = optim.Adam(unet.parameters(), lr=lr)
    # define the loss function
    loss = _get_loss_function(loss_function, chan_weights, 
                              device, losses, chan_losses)
    # get the dictionary that will be converted to a csv of losses
    #   contains columns for each channel, as we record channel-wise
    #   BCE loss in addition to the loss used for backprop
    channels = _index_channels_if_none(channels, x) 
    loss_dict = _get_loss_dict(channels)
    v_loss = _get_loss_function(loss_function, chan_weights, 
                                device, losses, chan_losses)
    validation_dict = {
        'epoch' : [], 
        'validation_loss' : [], 
        'data_id' : [], 
        'batch_id': []
    }
    if validate:
        no_iter = (epochs * len(x)) + ((epochs + 1) * len(vx))
    else:
        no_iter = epochs * len(x)
    # print the training into and log if applicable 
    bce_weights = _bce_weights(loss) # gets weights if using WeightedBCE
    _print_train_info(loss_function, bce_weights, epochs, lr, 
                      weights_are, device_name, out_dir, log, 
                      chan_losses, losses, channels, fork_channels)
    # loop over training data 
    v_y_hats = _train_loop(no_iter, epochs, x, y, ids, device, unet, 
                           out_dir, optimiser, loss, loss_dict,  
                           validate, vx, vy, vids, validation_dict, 
                           v_loss, update_every, log, name, channels, 
                           save_output)
    if save_output:
        print('Saving Final Results...')
        unet_path = _save_final_results(unet, out_dir, name, ids, validate, 
                                        loss_dict, v_y_hats, vids, 
                                        validation_dict)
    #_plots(out_dir, name, loss_function, validate) # 2 leaked semaphore objects... pytorch x mpl??
    return unet, unet_path



def _plots(out_dir, name, loss_function, validate):
    l_path = os.path.join(out_dir, 'loss_' + name + '.csv')
    v_path = None
    if validate:
        vl_path = os.path.join(out_dir, 'validation-loss_' + name + '.csv')
    save_loss_plot(l_path, loss_function, v_path=vl_path, show=False)
    save_channel_loss_plot(l_path, show=False)



def _get_loss_function(loss_function, chan_weights, device, 
                       losses, chan_losses):
    # define the loss function
    if loss_function == 'BCELoss':
        loss = nn.BCELoss()
    elif loss_function == 'DiceLoss':
        loss = DiceLoss()
    elif loss_function == 'WeightedBCE':
        loss = WeightedBCELoss(chan_weights=chan_weights, device=device)
    #elif loss_function == 'BCECentrenessPenalty':
       # loss = BCELossWithCentrenessPenalty()
    elif loss_function == 'EpochWeightedBCE':
        loss = EpochwiseWeightedBCELoss(weights_list=chan_weights, device=device)
    elif loss_function == 'Channelwise':
        loss = ChannelwiseLoss(losses, chan_losses, device)
    elif loss_function == 'MSELoss':
        loss = nn.MSELoss()
    else:
        m = 'Valid loss options are BCELoss, WeightedBCE, and DiceLoss'
        raise ValueError(m)
    return loss


def _load_weights(weights, unet):
    weights_are = 'naive'
    if weights is not None:
        unet.load_state_dict(weights)
        weights_are = 'pretrained'
    return weights_are


def _index_channels_if_none(channels, x):
    if channels is None:
        new_chans = ['channel_' + str(i) for i in range(x[0].shape[1])]
        return tuple(new_chans)
    else:
        return channels


def _get_loss_dict(channels):
    loss_dict = {'epoch' : [], 
                 'batch_num' : [], 
                 'loss' : [], 
                 'data_id' : []}
    for c in channels:
        loss_dict[c] = []
    return loss_dict


def _bce_weights(loss):
    bce_weights = None
    try:
        bce_weights = loss.chan_weights.data
    except:
        pass
    return bce_weights


def _print_train_info(loss_function, bce_weights, epochs, lr, 
                     weights_are, device_name, out_dir, log, 
                     chan_losses, losses, channels, fork_channels):
    s = LINE + '\n' + f'Loss function: {loss_function} \n'
    if bce_weights is not None:
        s = s + f'    Loss function channel weights: {bce_weights} \n'
    if losses is not None:
        chan_bd = []
        for c in chan_losses:
            if isinstance(c, slice):
                chan_bd.append(f'[{c.start}, {c.stop})')
            else:
                chan_bd.append(str(c))
        for i, l in enumerate(losses):
            s = s + f'    Loss for channels {chan_bd[i]}: {l}\n'
    s = s + 'Optimiser: Adam \n' + f'Learning rate: {lr} \n'
    s = s + LINE + '\n' 
    s = s + f'Training {weights_are} U-net for {epochs} '
    s = s + 'epochs with batch size 1 \n'
    s = s + f'Device: {device_name} \n'
    if channels is not None:
        s = s + f'Channels: {channels}\n'
    if fork_channels is not None:
        s = s + f'Channels per fork (according to channel order): {fork_channels}\n' 
    s = s + LINE
    print(s)
    if log:
        write_log(LINE, out_dir)
        write_log(s, out_dir)



def _train_loop(no_iter, epochs, x, y, ids, device, unet, out_dir,
                optimiser, loss, loss_dict,  validate, vx, vy, 
                vids, validation_dict, v_loss, update_every, log, 
                name, channels, save_output):
    v_y_hats = None
    # loop over training data 
    print_mem_every = 3 * update_every
    unet = unet.to(device=device, dtype=torch.float32)
    with tqdm(total=no_iter, desc='unet training') as progress:
        for e in range(epochs):
            _set_epoch_if_epoch_weighted(loss, e)
            if validate and e == 0:
                _set_epoch_if_epoch_weighted(v_loss, e, verbose=False)
                if e == 0:
                    # first validation at the start of the first epoch
                    v_y_hats = _validate(vx, vy, vids, 
                                         device, unet, v_loss, 
                                         progress, log, out_dir, 
                                         validation_dict, e, 0)
            running_loss = 0.0
            for i in range(len(x)):
                l = _train_step(i, x, y, ids, device, unet, optimiser, 
                                loss, loss_dict, e, channels)
                optimiser.step()
                running_loss += l.item()
                progress.update(1)
                if i % update_every == (update_every - 1):
                    s = f'Epoch {e} - running loss: ' 
                    s = s + f'{running_loss / update_every}'
                    print(s)
                    if log:
                        write_log(s, out_dir)
                    running_loss = 0.0
                #if i % print_mem_every == (print_mem_every - 1):
                    #print(torch.cuda.memory_summary())
            if validate:
                # validation at the end of the epoch
                batch_no = ((e + 1) * len(x))
                v_y_hats = _validate(vx, vy, vids, 
                                     device, unet, v_loss, 
                                     progress, log, out_dir, 
                                     validation_dict, e, batch_no)
            if save_output:
                print('Saving Training Checkpoint...')
                _save_checkpoint(unet.state_dict(), out_dir, 
                             f'{name}_epoch-{e}')  
            #torch.cuda.empty_cache()
    return v_y_hats


def _set_epoch_if_epoch_weighted(loss, e, verbose=True):
    if isinstance(loss, EpochwiseWeightedBCELoss):
        loss.current_epoch = e
        if verbose:
            print(f'Channel weights set to {loss.current_weights.data} for epoch {e} ')


def _train_step(i, xs, ys, ids, device, unet, optimiser, 
                loss, loss_dict, e, channels):
    x = load_tensor_from_zarr(i, xs)
    y = load_tensor_from_zarr(i, ys)
    x, y = _prep_x_y(x, y, device)
    optimiser.zero_grad()
    y_hat = unet(x.float())
    l = loss(y_hat, y)
    l.backward()
    optimiser.step()
    loss_dict['epoch'].append(e) 
    loss_dict['batch_num'].append(i)
    loss_dict['loss'].append(l.item())
    loss_dict['data_id'].append(ids[i])
    channel_losses_to_dict(y_hat, y, channels, loss, loss_dict)
    #y_hat = y_hat.detach().cpu().numpy()
    #y_hats.append(y_hat)
    #del y_hat
    del x, y, y_hat
    return l


def _validate(v_xs, v_ys, v_ids, device, unet, v_loss, progress, 
             log, out_dir, validation_dict, e, batch_no):
    validation_loss = 0.0
    with torch.no_grad():
        v_y_hats = []
        for i in range(len(v_xs)):
            v_x = load_tensor_from_zarr(i, v_xs)
            v_y = load_tensor_from_zarr(i, v_ys)
            v_x, v_y = _prep_x_y(v_x, v_y, device)
            v_y_hat = unet(v_x.float())
            v_y_hats.append(v_y_hat)
            vl = v_loss(v_y_hat, v_y)
            validation_loss += vl.item()
            validation_dict['epoch'].append(e)
            validation_dict['validation_loss'].append(vl.item())
            validation_dict['data_id'].append(v_ids[i])
            validation_dict['batch_id'].append(batch_no)
            progress.update(1)
        score = validation_loss / len(v_xs)
        s = f'Epoch {e} - validation loss: {score}'
        print(s)
        if log:
            write_log(s, out_dir)
    return v_y_hats


def _prep_x_y(x, y, device):
    x, y = torch.unsqueeze(x, 0), torch.unsqueeze(y, 0)
    x = torch.unsqueeze(x, 0)
    x, y = x.to(device), y.to(device)
    y = y.type(torch.float32)
    return x, y


def _save_final_results(unet, out_dir, name, ids, validate,
                        loss_dict, v_y_hats, v_ids, validation_dict):
    unet_path = _save_checkpoint(unet.state_dict(), out_dir, name, r=True)
    #_save_output(y_hats, ids, out_dir)
    loss_df = pd.DataFrame(loss_dict)
    loss_df.to_csv(os.path.join(out_dir, 'loss_' + name + '.csv'))
    if validate:
        _save_output(v_y_hats, v_ids, out_dir, name='_validation')
        v_loss_df = pd.DataFrame(validation_dict)
        v_loss_df.to_csv(os.path.join(out_dir, 
                         'validation-loss_' + name + '.csv'))
    return unet_path


def _save_checkpoint(checkpoint, out_dir, name, r=False):
    now = datetime.now()
    d = now.strftime("%y%d%m_%H%M%S")
    name = d + '_unet_' + name + '.pt'
    path = os.path.join(out_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    torch.save(checkpoint, path)
    if r:
        return path


def _save_output(y_hats, ids, out_dir, name=''):
    assert len(y_hats) == len(ids)
    os.makedirs(out_dir, exist_ok=True)
    for i in range(len(y_hats)):
        n = ids[i] + name +'_output.tif'
        p = os.path.join(out_dir, n)
        with TiffWriter(p) as tiff:
            tiff.write(y_hats[i].detach().cpu().numpy())
