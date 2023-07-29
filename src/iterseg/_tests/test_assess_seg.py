from iterseg._dock_widgets import _assess_segmentation, read_data
import os 
from pathlib import Path


FILE_PATH = __file__
ITERSEG_PATH = Path(FILE_PATH).parents[1]
DATA_PATH = os.path.join(ITERSEG_PATH, 'data')

# -------------------
# Load data in frames
# -------------------

#data_path_0 = os.path.join(DATA_PATH, 'data_in_frames')
data_path_1 = os.path.join(DATA_PATH, 'labels_in_frames')
seg = read_data(data_path_1, 'individual frames')
data_path_2 = os.path.join(DATA_PATH, 'GT_in_frames')
gt = read_data(data_path_2, 'individual frames')

save_path = os.path.join(DATA_PATH, 'temp')
_assess_segmentation(ground_truth=gt, model_segmentation=seg, save_dir=save_path, name='230728')