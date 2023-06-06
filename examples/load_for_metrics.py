import napari
from iterseg._dock_widgets import _load_data

# ABI MACBOOK
#images = '/Users/amcg0011/Data/pia-tracking/training_gt/images'
#ground_truth = '/Users/amcg0011/Data/pia-tracking/training_gt/ground_truth'
# /home/abigail/data/platelet_ground_truth/ground_truth

# DL MACHINE
images = '/home/abigail/data/platelet_ground_truth/images'
ground_truth = '/home/abigail/data/platelet_ground_truth/ground_truth'
model_1_u = '/home/abigail/data/iterseg/220209_iterseg_1/220209_152404_train-unet/my-unet/220902_152919_unet_my-unet.pt'
model_2_u = '/home/abigail/data/iterseg/220209_iterseg_2/220209_163559_train-unet/gl-unet/220902_164057_unet_gl-unet.pt'
model_1 = '/home/abigail/data/iterseg/220209_iterseg_1/model-1_labels-prediction.zarr'
model_2 = '/home/abigail/data/iterseg/220209_iterseg_2/model-2_labels-prediction.zarr'

viewer = napari.Viewer()
qtwidget, widget = viewer.window.add_plugin_dock_widget(
    'iterseg', 'load_data'
)

_load_data(viewer, data_path=images, data_type='individual frames', 
                layer_name='images', layer_type='Image', scale=(4, 1, 1))
_load_data(viewer, data_path=ground_truth, data_type='individual frames', 
                layer_name='ground truth', layer_type='Labels', scale=(4, 1, 1))
_load_data(viewer, data_path=model_1, data_type='image stacks', 
                layer_name='model 1', layer_type='Labels', scale=(4, 1, 1))
_load_data(viewer, data_path=model_2, data_type='image stacks', 
                layer_name='model 2', layer_type='Labels', scale=(4, 1, 1))


napari.run()

# /home/abigail/data/iterseg/220209_iterseg_1