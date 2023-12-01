from .helpers import get_regex_images
import numpy as np
from skimage.morphology._util import _offsets_to_raveled_neighbors
import torch
import torch.nn as nn


def channel_losses_to_dict(inputs, targets, channels, loss, loss_dict):
    #if targets.max() < 1 and targets.min() > 0:
    #loss = nn.BCELoss()
    #else:
       # loss = nn.MSELoss()
    for i, c in enumerate(channels):
        c_input = inputs[:, i, ...].detach()
        c_target = targets[:, i, ...].detach()
        l = loss(c_input, c_target)
        loss_dict[c].append(l.item())


# --------------
# Loss Functions
# --------------

class DiceLoss(nn.Module):
    '''
    DiceLoss: 1 - DICE coefficient 

    Adaptations: weights output channels equally in final loss. 
    This is necessary for anisotropic data.
    '''
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, channel_dim=1, smooth=1):
        '''
        inputs: torch.tensor
            Network predictions. Float
        targets: torch.tensor
            Ground truth labels. Float
        channel_dim: int
            Dimension in which output channels can be found.
            Loss is weighted equally between output channels.
        smooth: int
            Smoothing hyperparameter.
        '''
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs) 
        inputs, targets = flatten_channels(inputs, targets, channel_dim)
        intersection = (inputs * targets).sum(-1) 
        dice = (2.*intersection + smooth)/(inputs.sum(-1) + targets.sum(-1) + smooth) 
        loss = 1 - dice 
        return loss.mean()


class WeightedBCELoss(nn.Module):
    def __init__(
            self, 
            chan_weights, 
            device,
            reduction='mean', 
            final_reduction='mean', 
            channel_dim=1,
        ):
        super(WeightedBCELoss, self).__init__()
        self.bce = nn.BCELoss(reduction='none')
        self.chan_weights = torch.tensor(list(chan_weights)).to(device)
        self.reduction = reduction
        self.final_reduction = final_reduction
        self.channel_dim = channel_dim
        self.device = device


    def forward(self, inputs, targets):
        loss = weighted_BCE_loss(
            inputs, 
            targets, 
            self.chan_weights, 
            self.device,
            self.channel_dim, 
            self.reduction, 
            self.final_reduction
        )
        return loss



class EpochwiseWeightedBCELoss(nn.Module):
    def __init__(
            self, 
            weights_list, 
            device, 
            reduction='mean', 
            final_reduction='mean', 
            channel_dim=1,
        ):
        super(EpochwiseWeightedBCELoss, self).__init__()
        self._weights = torch.tensor(weights_list, requires_grad=False).to(device)
        self._reduction = reduction
        self._final_reduction = final_reduction
        self._channel_dim = channel_dim
        self._current_epoch = None
        self.current_weights = None
        self.device = device 

    
    @property
    def current_epoch(self):
        return self._current_epoch
    

    @current_epoch.setter
    def current_epoch(self, e):
        self._current_epoch = e
        self.current_weights = self._weights[e]


    def forward(self, inputs, targets):
        loss = weighted_BCE_loss(
            inputs, 
            targets, 
            self.current_weights, 
            self.device,
            self._channel_dim, 
            self._reduction, 
            self._final_reduction
        )
        return loss


class ChannelwiseLoss(torch.nn.Module):
    def __init__(self, losses, channels, device, channel_dim=1, 
                 ndims=5, reduction='mean'):
        super(ChannelwiseLoss, self).__init__()
        self.losses = [loss.to(device) for loss in losses]
        self.reduction = reduction
        slices = []
        for c in channels:
            s_ = [slice(None, None),] * ndims
            s_[channel_dim] = c
            s_ = tuple(s_)
            slices.append(s_)
        self.slices = slices

    def forward(self, inputs, targets):
        losses = torch.zeros(len(self.losses))
        for i, s_ in enumerate(self.slices):
            ipt = inputs[s_]
            tgt = targets[s_]
            loss = self.losses[i](ipt, tgt)
            losses[i] = loss
        if self.reduction == 'mean':
            loss = losses.mean()
        elif self.reduction == 'sum':
            loss = losses.sum()
        else:
            raise ValueError(f'{self.reduction} is not a valid reduction for this loss')
        return loss


def weighted_BCE_loss(
        inputs, 
        targets, 
        chan_weights, 
        device,
        channel_dim=1, 
        reduction='mean', 
        final_reduction='mean'
    ):
    bce = nn.BCELoss(reduction='none').to(device)
    inputs, targets = flatten_channels(inputs, targets, channel_dim)
    unreduced = bce(inputs, targets)
    if reduction == 'mean':
        channel_losses = unreduced.mean(-1) * chan_weights
    elif reduction == 'sum':
        channel_losses = unreduced.sum(-1) * chan_weights
    else:
        raise ValueError('reduction param must be mean or sum')
    if final_reduction == 'mean':
        loss = channel_losses.mean()
    elif final_reduction == 'sum':
        loss = channel_losses.sum()
    else:
        raise ValueError('final_reduction must be mean or sum')
    return loss


class SegmentationLoss(nn.Module):

    '''
    Here's the thing... this is super difficult because it can't be used to train a network
    with backprop (i.e., segmentation process isn't differentiable). Would require another
    means of training weights.
    '''
    def __init__(self, GT_dir, ids, seg_func, seg_kwargs):
        super(SegmentationLoss, self).__init__()
        self._func = seg_func 
        self._kwargs = seg_kwargs
        self._batch_no = None
        # get the GTs in correct order
        self._GT_stack = get_regex_images(GT_dir, r'\d{6}_\d{6}_\d{1,3}_GT.tif', ids)


    def forward(self, inputs, targets):
        pass


    def segment(self, data):
        return self._func(data, **self._kwargs)


    def get_GT(self):
        array = self._GT_stack[self._batch_no].compute()
        return array


    @property
    def batch_no(self):
        return self._batch_no


    @batch_no.setter
    def batch_no(self, i):
        self._batch_no = i


# ----------------
# Helper Functions 
# ----------------

def flatten_channels(inputs, targets, channel_dim):
    '''
    Helper function to flatten inputs and targets for each channel

    E.g., (1, 3, 10, 256, 256) --> (3, 655360)

    Parameters
    ----------
    inputs: torch.Tensor
        U-net output
    targets: torch.Tensor
        Target labels
    channel_dim: int
        Which dim represents output channels? 
    '''
    order = [channel_dim, ]
    for i in range(len(inputs.shape)):
        if i != channel_dim:
            order.append(i)
    inputs = inputs.permute(*order)
    inputs = torch.flatten(inputs, start_dim=1)
    targets = targets.permute(*order)
    targets = torch.flatten(targets, start_dim=1)
    return inputs, targets

  
# -------------------
# Probably Not Useful
# -------------------
 
class BCELossWithCentrenessPenalty(nn.Module):

    def __init__(
                 self, 
                 weight=0.1
                 ):
        super(BCELossWithCentrenessPenalty, self).__init__()
        self.weight = weight
        self.bce = nn.BCELoss()

    
    def forward(self, inputs, targets):
        loss = self.bce(inputs, targets)
        affinities = inputs[:3, ...].detach()
        centreness = targets[3, ...].detach()
        penalty = self.centreness_penalty(affinities, centreness)
        loss = loss.subtract(self.weight * penalty)
        return loss


    def centreness_penalty(self, affinities, centreness):
        '''
        Parameters
        ----------
        affinities: torch.Tensor
            shape (3, z, y, x)
        centreness: torch.Tensor
            shape (z, y, x)
        '''
        scores = []
        for i in range(affinities.shape[0]):
            aff = affinities[i].detach()
            score = self.bce(aff, centreness)
            scores.append(score)
        scores = torch.tensor(scores)
        score = scores.mean()
        return score



class CentroidLoss(nn.Module):

    def __init__(
                 self, 
                 selem=np.ones((3, 9, 9)), 
                 centre=(4, 4, 4), 
                 chan_weights=(1., 1., 1.), 
                 eta = 0.5,
                 phi = 0.5
                 ):
        super(WeightedBCELoss, self).__init__()
        self.selem = selem
        self.centre = centre
        self.shape = None
        self.BCELoss = WeightedBCELoss(chan_weights=chan_weights)
        self.eta = eta
        self.phi = phi


    def forward(
                self, 
                inputs, 
                targets, 
                selem=np.ones((3, 9, 9)), 
                centre=(1, 4, 4), 
                channel_dim=1
                ):
        
        # Prep, because flat is easier
        block_shape = [s for s in inputs.shape[-3:]]
        block_shape = tuple(block_shape)
        inputs, targets = flatten_channels(inputs, targets, self.channel_dim)

        # -----------------------
        # BCE Loss for Affinities
        # -----------------------
        # ???? pytorch will make this unexpectedly difficult I'm sure
        input_affs = inputs[:3] # pretty sure this won't work LOL
        target_affs = targets[:3]
        loss = self.BCELoss(input_affs, target_affs)

        # -------------------
        # Centroid Similarity
        # -------------------
        target_cent = targets[3].numpy()
        # penalise similarity of affinities to centroids - high score
        aff_penalties = []
        for i in range(3):
            aff_chan = inputs[i].numpy()
            penalty = self.centroid_similarity(aff_chan, target_cent)
        aff_penalties = np.array(aff_penalties)
        aff_penalty = aff_penalties.mean() * self.phi
        aff_penalties = torch.from_numpy(aff_penalty)
        # penalise difference of centroid output channel from centroids
        input_cent = inputs[3].numpy()
        cent_penalty = (1 - self.centroid_similarity(input_cent, target_cent)) * self.eta
        cent_penalty = torch.from_numpy(cent_penalty)
        # add penalties to 
        loss.add_(aff_penalty)
        loss.add_(cent_penalty)
        return loss



    def centroid_similarity(
                            self, 
                            inputs, 
                            targets,
                            shape 
                            ):
        """
        Metric for similarity to centroid. Computes a similarity based on
        inverse of normaised euclidian distance for every centroid neighbor.
        Neighbors are determined by a structing element (cell-ish dims)

        The output should be between 0-1 (therefore easily invertable)
        MAKE SURE THIS IS TRUE!!!

        Notes
        -----
        Currently uses a structing element to find centroid neighbors.
        Hoping to add an option for using segmentation to get neighbors. 

        """
        offsets = _offsets_to_raveled_neighbors(shape, 
                                                self.selem, 
                                                self.centre)

        euclid_dists = self.euclidian_distances()
        weights = euclid_dists - 1
        centroids = np.argwhere(targets == 1.)
        score = 0
        for c in centroids:
            max_ind = inputs.shape[-1]
            raveled_indices = c + offsets
            in_bounds_indices = np.array([idx for idx in raveled_indices \
                                            if idx >= 0 and idx < max_ind])
            neighbors = inputs[in_bounds_indices]
            weighted = neighbors * weights
            score += weighted.mean()
        return score


    def euclidian_distances(self):
        '''
        Compute euclidian distances of each index from 
        '''
        selem_indices = np.stack(np.nonzero(self.selem), axis=-1)
        distances = []
        centre = np.array(self.centre)
        for ind in selem_indices:
            dist = np.linalg.norm(centre - ind)
            distances.append(dist)
        distances = np.array(distances)
        return distances / distances.max()

