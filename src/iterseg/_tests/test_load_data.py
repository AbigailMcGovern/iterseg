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

# images in 3D frames
# -------------------
# tif files and zarr
data_path_0 = os.path.join(DATA_PATH, 'data_in_frames')
_load_data(v, directory=data_path_0, data_type='individual frames', 
           layer_name='test images', layer_type='Image', 
           scale=(4, 1, 1), translate=(0, 0, 0))



# labels in 3D frames
# -------------------
# zarr files
data_path_1 = os.path.join(DATA_PATH, 'labels_in_frames')
_load_data(v, directory=data_path_1, data_type='individual frames', 
           layer_name='test segmentations', layer_type='Labels', 
           scale=(4, 1, 1), translate=(0, 0, 0))


data_path_2 = os.path.join(DATA_PATH, 'GT_in_frames')
_load_data(v, directory=data_path_2, data_type='individual frames', 
           layer_name='test gt', layer_type='Labels', 
           scale=(4, 1, 1), translate=(0, 0, 0))



# individual 3D zarr
# ------------------
data_path_3 = os.path.join(data_path_0, 'saline_example.zarr')
_load_data(v, data_file=data_path_3, data_type='individual frames', 
           layer_name='saline', layer_type='Image', 
           scale=(4, 1, 1), translate=(0, 0, 0))



# individal 3D tif
# ----------------
data_path_4 = os.path.join(data_path_0, '191016_IVMTR12_Inj4_cang_exp3_fr125.tif')
_load_data(v, data_file=data_path_4, data_type='individual frames', 
           layer_name='cangrelor', layer_type='Image', 
           scale=(4, 1, 1), translate=(0, 0, 0))



# 4D stacks - zarr and tiff
# -------------------------



napari.run()

# -------------------
# Load data in stacks
# -------------------

#TODO