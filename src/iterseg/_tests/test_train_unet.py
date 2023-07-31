from genericpath import exists
import napari
from iterseg.train import train_unet
from iterseg._dock_widgets import _train_from_viewer
import numpy as np
from pathlib import Path
from shutil import rmtree
import os

FILE_PATH = __file__
ITERSEG_PATH = Path(FILE_PATH).parents[1]
DATA_PATH = os.path.join(ITERSEG_PATH, 'data')
OUT_PATH = os.path.join(DATA_PATH, 'temp')


xs = [np.random.random((10, 256, 256)) for _ in range(15)]
vxs = [np.random.random((10, 256, 256)) for _ in range(5)]
ys = [np.random.random((5, 10, 256, 256)) for _ in range(15)]
vys = [np.random.random((5, 10, 256, 256)) for _ in range(5)]


def test_train_unet_BCELoss_save():
    outdir = os.path.join(DATA_PATH, 'temp')
    os.makedirs(outdir, exist_ok=True)
    name = 'test_train'
    u = train_unet(xs, vxs, ys, vys, 
                   out_dir=outdir, name=name, epochs=1)


def test_train_unet_DICE():
    u = train_unet(xs, vxs, ys, vys, epochs=1, 
                   loss_function='DICELoss')


def test_train_unet_WeightedBCE():
    u = train_unet(xs, vxs, ys, vys, epochs=1, 
                   loss_function='WeightedBCE', 
                   chan_weights=(2., 1., 1., 1., 2.))


def test_train_pipeline():
    images = [np.random.random((33, 512, 512)) for _ in range(2)]
    ground_truth = [np.random.choice(a=[True, False], size=(33, 512, 512)) for _ in range(2)]
    v = napari.Viewer()
    for image in images:
        v.add_image(image, scale=(4, 1, 1))
    for gt in ground_truth:
        v.add_labels(gt, scale=(4, 1, 1))
    _train_from_viewer(v, v.layers['images'], v.layers['ground_truth'], OUT_PATH,
                       n_each=5, epochs=1, predict_labels=False,)