from iterseg._dock_widgets import _load_data
from pathlib import Path
import os
import napari

FILE_PATH = __file__
ITERSEG_PATH = Path(FILE_PATH).parents[1]
DATA_PATH = os.path.join(ITERSEG_PATH, 'data')

# -------------------
# Load data in frames
# -------------------

v = napari.Viewer()

data_path_0 = os.path.join(DATA_PATH, 'data_in_frames')
_load_data(v, data_path=data_path_0, data_type='individual frames', 
           layer_name='test images', layer_type='Image', 
           scale=(4, 1, 1), translate=(0, 0, 0))


data_path_1 = os.path.join(DATA_PATH, 'labels_in_frames')
_load_data(v, data_path=data_path_1, data_type='individual frames', 
           layer_name='test segmentations', layer_type='Labels', 
           scale=(4, 1, 1), translate=(0, 0, 0))


data_path_2 = os.path.join(DATA_PATH, 'GT_in_frames')
_load_data(v, data_path=data_path_2, data_type='individual frames', 
           layer_name='test gt', layer_type='Labels', 
           scale=(4, 1, 1), translate=(0, 0, 0))


napari.run()

# -------------------
# Load data in stacks
# -------------------

#TODO