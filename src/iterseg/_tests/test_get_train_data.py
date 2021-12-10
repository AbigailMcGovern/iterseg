from iterseg.train_io import get_train_data
import numpy as np

images = [np.random.random((33, 512, 512)) for _ in range(2)]
ground_truth = [np.random.choice(a=[True, False], size=(33, 512, 512)) for _ in range(2)]

def test_z1_y1_x1_m_cl():
    scale = [(4, 1, 1), (4, 1, 1)]
    n_each = 10
    channels = ('z-1', 'y-1', 'x-1', 'mask', 'centreness-log')
    train_dict = get_train_data(images, ground_truth, n_each=n_each, channels=channels)


def test_z1_y1_x1_m_c():
    scale = [(4, 1, 1), (4, 1, 1)]
    n_each = 10
    channels = ('z-1', 'y-1', 'x-1', 'mask', 'centreness')
    train_dict = get_train_data(images, ground_truth, n_each=n_each, channels=channels)


def test_z1_y1_x1_m_cg():
    scale = [(4, 1, 1), (4, 1, 1)]
    n_each = 10
    channels = ('z-1', 'y-1', 'x-1', 'mask', 'centroid-gauss')
    train_dict = get_train_data(images, ground_truth, n_each=n_each, channels=channels)