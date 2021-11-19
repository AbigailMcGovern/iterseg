from pydantic.errors import TupleError
from .training_experiments import get_experiment_dict, run_experiment
import napari
from magicgui import magic_factory, magicgui
from datetime import datetime



@magic_factory(
    call_button=True, 
    mask_prediction={'choices': ['mask', 'centreness']}, 
    centre_prediciton={'choices': ['centreness-log', 'centreness', 'centroid-gauss']},
    affinities_extent={'widget_type' : 'LiteralEvalLineEdit'},
    training_name={'widget_type': 'LineEdit'}, 
    loss_function={'choices': ['BCELoss', 'DiceLoss']}, 
    output_dir={'widget_type': 'LineEdit'}
    )
def train_unet(
    viewer: napari.viewer.Viewer, 
    mask_prediction='mask', 
    centre_prediciton='centreness-log', #lol btw this is a typo in the whole repo :P
    affinities_extent=1, 
    training_name='my-unet',
    loss_function='BCELoss', 
    learning_rate=0.01, 
    epochs=4,
    validation_prop=0.2, 
    n_each=50,
    output_dir='.'
    ):
    channels_list = construct_channels_list(affinities_extent, mask_prediction, 
                                        centre_prediciton)
    condition_name = [training_name, ]
    image_list = [l for l in viewer.layers if l.metadata.get('train_img') == True]
    labels_list = [viewer.layers[l.metadata['gt_partner']] for l in image_list]
    conditions_list = construct_conditions_list(image_list, loss_function, learning_rate, epochs)
    exp_dict = get_experiment_dict(channels_list, condition_name, 
                                   conditions_list=conditions_list, 
                                   validation_prop=validation_prop, 
                                   n_each=n_each)
    run_experiment(exp_dict, image_list, labels_list, output_dir)



def construct_channels_list(
    affinities_extent, 
    mask_prediction, 
    centre_predicition
    ):
    dims = ('z', 'y', 'x')
    affs = []
    if isinstance(affinities_extent, tuple):
        m = f'please ensure the length of the affinities extent tuple matches the number of dims in {dims}'
        assert len(affinities_extent) == len(dims), m
    elif isinstance(affinities_extent, int):
        affinities_extent = [affinities_extent, ] * len(dims)
        affinities_extent = tuple(affinities_extent)
    else:
        m = 'Please insert affinities extent of type tuple or int (e.g., 1 or (2, 2, 1))'
        raise TypeError(m)
    for i, d in enumerate(dims):
        n_affs = affinities_extent[i]
        for n in range(1, n_affs + 1):
            affs.append(f'{d}-{n}')
    affs.append(mask_prediction)
    affs.append(centre_predicition)
    affs = [tuple(affs), ]
    return affs


def construct_conditions_list(
    image_list, 
    loss_function, 
    learning_rate, 
    epochs
    ):
    scale = [tuple(l.scale) for l in image_list]
    condition_dict = {
        'scale' : scale, 
        'lr' : learning_rate, 
        'loss_function' : loss_function, 
        'epochs' : epochs
    }
    return [condition_dict, ]


# add metadata to image and GT pairs so that they can be identified and used for training
@magicgui(call_button=True)
def assign_train_data(
    image: napari.types.ImageData, 
    labels: napari.types.LabelsData
    ):
    image_meta = {'train_img' : True, 'gt_partner' : labels.name}
    image.metadata.update(image_meta)
    labels_meta = {'train_gt' : True, 'img_partner' : image.name}
    labels.metadata.update(labels_meta)

