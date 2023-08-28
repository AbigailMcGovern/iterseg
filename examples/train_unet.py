from iterseg._dock_widgets import _load_data, _train_from_viewer
import napari


od = '/Users/abigailmcgovern/Data/iterseg/kidney_development/training/230816'
ip = '/Users/abigailmcgovern/Data/iterseg/kidney_development/training/images'
gp = '/Users/abigailmcgovern/Data/iterseg/kidney_development/training/ground_truth'

v = napari.Viewer()

_load_data(v, directory=ip, data_type='individual frames', 
           layer_name='images', layer_type='Image', 
           scale=(4, 1, 1), translate=(0, 0, 0))

_load_data(v, directory=gp, data_type='individual frames', 
           layer_name='gt', layer_type='Labels', 
           scale=(4, 1, 1), translate=(0, 0, 0))


_train_from_viewer(v, v.layers['images'], v.layers['gt'], od, (4, 1, 1), training_name="kidneynet")