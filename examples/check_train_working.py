import napari

images = '/Users/amcg0011/Data/pia-tracking/training_gt/images'
ground_truth = '/Users/amcg0011/Data/pia-tracking/training_gt/ground_truth'

viewer = napari.Viewer()
qtwidget, load_widget = viewer.window.add_plugin_dock_widget(
    'iterseg', 'load_data'
)

load_widget.load_data(data_path=images, data_type='individual frames', 
                        layer_name='images', layer_type='Image', scale=(4, 1, 1))
load_widget.load_data(data_path=ground_truth, data_type='individual frames', 
                        layer_name='ground truth', layer_type='Labels', scale=(4, 1, 1))

img_layer = viewer.layers['images']
lab_layer = viewer.layers['ground truth']

qtwidget, train_widget = viewer.window.add_plugin_dock_widget(
    'iterseg', 'load_data'
)