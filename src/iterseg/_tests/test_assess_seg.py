from iterseg._dock_widgets import _assess_segmentation, read_data
import os 
from pathlib import Path


FILE_PATH = __file__
ITERSEG_PATH = Path(FILE_PATH).parents[1]
DATA_PATH = os.path.join(ITERSEG_PATH, 'data')


# --------------------
# Test 4D GT and 4D MR
# --------------------

# model result (4D)
data_path_1 = os.path.join(DATA_PATH, 'labels_in_frames')
seg, _ = read_data(data_path_1, None, 'individual frames')
# ground truth (4D)
data_path_2 = os.path.join(DATA_PATH, 'GT_in_frames')
gt, _ = read_data(data_path_2, None, 'individual frames')
# assess segmentation
save_path = os.path.join(DATA_PATH, 'temp')
_assess_segmentation(ground_truth=gt, model_segmentation=seg, save_dir=save_path, name='230728')



# --------------------
# Test 4D GT and 3D MR
# --------------------

# model result (3D)
data_path_3 = os.path.join(data_path_1, '210511_IVMTR105_Inj5_DMSO2_exp3_fr54_labels.zarr')
seg, _ = read_data(data_path_3, None, 'individual frames')
# ground truth (4D)
gt, _ = read_data(data_path_2, None, 'individual frames')
# assess segmentation
save_path = os.path.join(DATA_PATH, 'temp')
_assess_segmentation(ground_truth=gt, model_segmentation=seg, save_dir=save_path, name='4DGT_3DMR')


# --------------------
# Test 3D GT and 4D MR
# --------------------


# --------------------
# Test 3D GT and 3D MR
# --------------------