from heapq import heappop, heappush
from collections import namedtuple

from scipy import ndimage as ndi
import numba
import numpy as np
from skimage import filters, morphology, feature
from skimage.morphology._util import (
        _offsets_to_raveled_neighbors, _validate_connectivity
        )

# ---------
# Watershed
# ---------


def affinity_watershed(
        image, marker_coords, mask, scale=None, out=None, py_func=False
        ):
    image_raveled, marker_coords, offsets, mask, output = _prep_data(
            image, marker_coords, mask, output=out
            )
    if scale is not None:
        image_raveled *= np.abs(scale)
    if py_func:
        raveled_affinity_watershed.py_func(
                image_raveled, marker_coords, offsets, mask, output
                )
    else:
        raveled_affinity_watershed(
                image_raveled, marker_coords, offsets, mask, output
                )
    shape = image.shape[1:]
    output = output.reshape(shape)
    return output


def _prep_data(image, marker_coords, mask=None, output=None):
    # INTENSITY VALUES
    im_ndim = image.ndim - 1  # the first dim should represent affinities
    image_shape = image.shape[1:]
    raveled_image = np.zeros((image.shape[0], image[0].size),
                             dtype=image.dtype)
    for i in range(image.shape[0]):
        raveled_image[i] = image[i].ravel()
    # NEIGHBORS
    selem, centre = _validate_connectivity(im_ndim, 1, None)
    # array of shape (ndim * 2, 2) giving the indicies of neighbor affinities
    offsets = _indices_to_raveled_affinities(image_shape, selem, centre)
    raveled_markers = np.apply_along_axis(
            _raveled_coordinate, 1, marker_coords, shape=image_shape
            )
    if mask is None:
        small_shape = [s - 2 for s in image_shape]
        mask = np.ones(small_shape, dtype=bool)
        mask = np.pad(mask, 1, constant_values=0)
        assert image_shape == mask.shape
    mask_raveled = mask.ravel()
    if output is None:
        output = np.zeros(mask_raveled.shape, dtype=np.int32)
    labels = np.arange(len(raveled_markers)) + 1
    output[raveled_markers] = labels
    return (raveled_image, raveled_markers, offsets, mask_raveled, output)


def _raveled_coordinate(coordinate, shape):
    # array[z, y, x] = (
    #     array.ravel()[
    #         z * array.shape[1] * array.shape[2]
    #         + y * array.shape[2]
    #         + x
    #     ]
    # )
    raveled_coord = 0
    for i in range(len(coordinate)):
        to_add = coordinate[i]
        for j in range(len(shape)):
            if j > i:
                to_add *= shape[j]
        raveled_coord += to_add
    return raveled_coord


def _indices_to_raveled_affinities(image_shape, selem, centre):
    im_offsets = _offsets_to_raveled_neighbors(image_shape, selem, centre)
    # im_offsets[-len(image_shape):] = 0
    affs = np.concatenate([
            np.arange(len(image_shape)),
            np.arange(len(image_shape))[::-1]
            ])
    indices = np.stack([affs, im_offsets], axis=1)
    return indices


@numba.jit(cache=True, nopython=False)
def raveled_affinity_watershed(
        image_raveled, marker_coords, offsets, mask, output
        ):
    """Compute affinity watershed on raveled arrays.

    Parameters
    ----------
    image_raveled : 2D array of float(32), shape (npixels, ndim)
        The z, y, and x affinities around each pixel.
    marker_coords : 1D array of int
        The location of each marker along the pixels dimension of
        ``image_raveled``.
    offsets : 1D array of int
        The signed offsets to each neighboring pixel.
    mask : 1D array of bool, shape (npixels,)
        True for pixels to which the watershed should spread.
    output : 1D array of int
        The output array for markers.
    """
    n_neighbors = offsets.shape[0]
    age = 0
    marker_coords = marker_coords.astype(np.intp)
    offsets = offsets.astype(np.intp)
    aff_offsets = offsets.copy().astype(np.intp)
    aff_offsets[:int(len(offsets) / 2), 1] = 0
    heap = [
            Element(
                    image_raveled[0, 0], age, marker_coords[0],
                    marker_coords[0]
                    )
            ]
    _ = heappop(heap)
    # add each seed to the stack
    for i in range(marker_coords.shape[0]):
        index = marker_coords[i]
        value = np.float32(0.)
        source = index
        index = index
        elem = Element(value, age, index, source)
        heappush(heap, elem)
    # remove from stack until empty
    while len(heap) > 0:
        elem = heappop(heap)
        for i in range(n_neighbors):
            # get the flattened address of the neighbor
            # offsets are 2d (size, 2) with columns 0 and 1 corresponding to
            # affinities (ie axis) and image neighbour indices respectively
            neighbor_index = elem.index + offsets[i, 1]
            if not mask[neighbor_index]:
                # neighbor is not in mask, move on to next neighbor
                continue
            if output[neighbor_index]:
                # if there is a non-zero value in output, move on to next
                # neighbor
                continue
            # if the neighbor is in the mask and not already labeled,
            # label it then add it to the queue
            output[neighbor_index] = output[elem.index]
            value = image_raveled[aff_offsets[i, 0],
                                  aff_offsets[i, 1] + elem.index]
            age += 1
            new_elem = Element(value, age, neighbor_index, elem.source)
            heappush(heap, new_elem)
    return output


Element = namedtuple('Element', ['value', 'age', 'index', 'source'])


def segment_output_image(
        unet_output,
        affinities_channels,
        centroids_channel,
        thresholding_channel,
        scale=None,
        absolute_thresh=None,
        out=None,
        py_func=False,
        ):
    '''
    Parameters
    ----------
    unet_output: np.ndarray or dask.array.core.Array
        Output from U-net inclusive of all channels. If there is an extra
        dim of size 1, this will be squeezed out. Therefore shape may be
        (1, c, z, y, x) or (c, z, y, x).
    affinities_channels: tuple of int
        Ints, in order (z, y, x) describe the channel indicies to which
        the z, y, and x short-range affinities belong.
    centroids_channel: int
        Describes the channel index for the channel that is used to find
        centroids.
    thresholding_channel: in
        Describes the channel index for the channel that is used to find
        the mask for watershed.
    '''
    unet_output = np.asarray(np.squeeze(unet_output))
    # Get the affinities image (a, z, y, x)
    affinities = unet_output[list(affinities_channels)]
    affinities /= np.max(affinities, axis=(1, 2, 3)).reshape((-1, 1, 1, 1))
    affinities = np.pad(
            affinities,
            ((0, 0), (1, 1), (1, 1), (1, 1)),
            mode='constant',
            constant_values=0,
            )
    # Get the image for finding centroids
    centroids_img = unet_output[centroids_channel]
    # find the centroids
    centroids = _get_centroids(centroids_img) + 1  # account for padding
    # Get the image for finding the mask
    masking_img = unet_output[thresholding_channel]
    # find the mask for use with watershed
    if absolute_thresh is None:
        mask = _get_mask(masking_img)
    else:
        mask = masking_img > absolute_thresh
    mask = np.pad(mask, 1, constant_values=0)  # edge voxels must be 0
    mask, centroids = _remove_unwanted_objects(
            mask, centroids, min_area=10, max_area=10000000
            )
    # affinity-based watershed
    segmentation = affinity_watershed(
            affinities, centroids, mask, scale=scale, out=out, py_func=py_func
            )
    segmentation = segmentation[1:-1, 1:-1, 1:-1]
    seeds = centroids - 1
    return segmentation, seeds, mask


def _get_mask(img, sigma=2):
    thresh = filters.threshold_otsu(filters.gaussian(img, sigma=sigma))
    mask = img > thresh
    return mask


def _get_centroids(cent, gaussian=True):
    if gaussian:
        cent = filters.gaussian(cent, sigma=(0, 1, 1))
    centroids = feature.peak_local_max(cent, threshold_abs=.04)
    return centroids


def _remove_unwanted_objects(mask, centroids, min_area=0, max_area=1000000):
    labels, _ = ndi.label(mask)
    labels_no_small = morphology.remove_small_objects(
            labels, min_size=min_area
            )
    labels_large = morphology.remove_small_objects(
            labels_no_small, min_size=max_area
            )
    labels_goldilocks = labels_no_small ^ labels_large
    centroid_labels = labels_goldilocks[tuple(centroids.T)]
    new_centroids = centroids[centroid_labels > 0]
    new_mask = labels_goldilocks.astype(bool)
    return new_mask, new_centroids


if __name__ == '__main__':
    from skimage import data, segmentation, morphology
    foreground = data.binary_blobs(length=128, n_dim=3, volume_fraction=0.35)
    centroids = ndi.distance_transform_edt(foreground)
    affz, affy, affx = [
            filters.scharr(foreground.astype(float), axis=i) for i in range(3)
            ]
    volume = np.stack([affz, affy, affx, centroids, foreground],
                      axis=0).astype(np.float32)
    import time
    start = time.time()
    _, _, _ = segment_output_image(
            volume, (0, 1, 2), 3, 4, absolute_thresh=0.5
            )
    end = time.time()
    print('warmup', end - start)

    start = time.time()
    labels, _, _ = segment_output_image(
            volume, (0, 1, 2), 3, 4, absolute_thresh=0.5
            )
    end = time.time()
    print('second run', end - start)

    start = time.time()
    labels_py, _, _ = segment_output_image(
            volume, (0, 1, 2), 3, 4, absolute_thresh=0.5, py_func=True
            )
    end = time.time()
    print('python', end - start)

    start = time.time()
    _ = segmentation.watershed(
            np.sqrt(np.sum(volume[:3]**2, axis=0)),
            ndi.label(morphology.local_maxima(centroids, indices=False))[0],
            mask=foreground,
            )
    end = time.time()
    print('skimage watershed', end - start)

    print('numba and python are equal', np.all(labels == labels_py))

    import napari
    napari.view_labels(labels, ndisplay=3)
    napari.run()
