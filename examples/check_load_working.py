import napari
from iterseg._dock_widgets import _load_data

images = '/Users/amcg0011/Data/pia-tracking/training_gt/images'
ground_truth = '/Users/amcg0011/Data/pia-tracking/training_gt/ground_truth'

viewer = napari.Viewer()
qtwidget, widget = viewer.window.add_plugin_dock_widget(
    'iterseg', 'load_data'
)

_load_data(viewer, data_path=images, data_type='individual frames', 
                layer_name='images', layer_type='Image', scale=(4, 1, 1))
_load_data(viewer, data_path=ground_truth, data_type='individual frames', 
                layer_name='ground truth', layer_type='Labels', scale=(4, 1, 1))