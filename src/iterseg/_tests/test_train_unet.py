from genericpath import exists
import napari
from iterseg.train import train_unet
from iterseg._train_from_napari import train_from_viewer
import numpy as np
from pathlib import Path
from shutil import rmtree
import os

CURRENT_PATH = Path(__file__).parent.resolve()
IS_PATH = CURRENT_PATH.parents[1]
DATA_PATH = str(IS_PATH / 'data')


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
    train_from_viewer(v, n_each=5, epochs=1)