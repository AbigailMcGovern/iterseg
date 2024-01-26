# iterseg

[![License](https://img.shields.io/pypi/l/iterseg.svg?color=green)](https://github.com/abigailmcgovern/iterseg/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/iterseg.svg?color=green)](https://pypi.org/project/iterseg)
[![Python Version](https://img.shields.io/pypi/pyversions/iterseg.svg?color=green)](https://python.org)
[![tests](https://github.com/abigailmcgovern/iterseg/workflows/tests/badge.svg)](https://github.com/abigailmcgovern/iterseg/actions)
[![codecov](https://codecov.io/gh/abigailmcgovern/iterseg/branch/main/graph/badge.svg)](https://codecov.io/gh/abigailmcgovern/iterseg)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/iterseg)](https://napari-hub.org/plugins/iterseg)

napari plugin for iteratively improving a deep learning-based unet-watershed segmentation. 

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/docs/plugins/index.html
-->

## Installation
Install iterseg using pip. Assuming you have python and pip installed (e.g., via miniconda), you can install iterseg with only one line, typed into terminal (MacOS/Linux) or annaconda prompt (Windows). We recomend installing into a [new environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#) as some of our dependencies may not play well in the sandpit with certain versions of packages that may exist in a prexisting one. 

```bash
pip install iterseg napari
```


## Opening iterseg
Once `iterseg` is installed, you can access it through the napari viewer, which you can open from the command line (e.g., terminal (MacOS), anaconda prompt (Windows), git bash (Windows), etc.). To open napari simply type into the command line:
```bash
napari
```

## Loading data
Once you've opened napari, you can load image, labels, or shapes data through the `load_data` widget. to open the widget go to **plugins/iterseg/load_data** at the top left of your screen (MacOS) or viewer (Windows). 

 ![find the widgets](https://github.com/AbigailMcGovern/iterseg/blob/main/docs/images/load_data_find.png)

Once the widget appears at the right of the napari window, enter the name you want to give the data you are loading (this will appear in the layers pannel on the left of the window). Choose the type of layer you want to load (Image, Labels, or Shapes: segmentations are loaded as labels layer). You can load a folder of files or a zarr file using "choose directory" (zarrs are recognised as a folder of files) or you can load a tiff file using "choose file". You can tell the program what the scale of the 3D frames will be in (in the format (z, y, x)).

 ![load data](https://github.com/AbigailMcGovern/iterseg/blob/main/docs/images/load_data.png)

If you are using a single image file (3D, 4D, 5D - ctzyx) or a directory of 3D images (zyx), for "data type" choose "individual frames". If you are using a directory of 4D images (tzyx) choose "image stacks". If you are loading a file that is 4D or 5D and want to load time points (4D: tzyx, czyx) or channels (5D: ctzyx) as individual layers, select "split channels". 

## Segmenting images

You can segment data using the "segment_data" widget, which can be found at **plugins/iterseg/segment_data**. Once the widget appears, you can choose (1) the image layer you want to segment, (2) the folder into which to save the data, (3) the name you want to give the output file, (4) the type of segmentation to use, (5: optionally) the path to a neural network or configuration file, (6: optionally) a layer produced during training which contains metadata pointing to the trained neual network, (7) chunk size (the size of the neural network input), (8) margin (the margin of overlap between chunks). There is also an optional tickbox for debugging. If this is selected, errors will be easier to identify but you won't be able to interact with the viewer until the segmentation is done. 

 ![segmentation in progress](https://github.com/AbigailMcGovern/iterseg/blob/main/docs/images/segmenting_in_progress.png)

  ![segmented data](https://github.com/AbigailMcGovern/iterseg/blob/main/docs/images/segmented_data.png)

Segmented images can be used to more quickly generate ground truth for training, to assess segmentation quality, or for downstream analyses. 

### Segmentation algorithms
#### Affinity U-Net Watershed
The affinity U-net watershed is a feature based instance segmentation algorithm. A trained U-net predicts an edge affinity graph (basically boundaries in the x, y, and z axes), a map of centre points, and a mask that specifies which pixesl belong to objects. The feature map is fed to a modified watershed algorithm. The object centres are used to find seeds for the watershed and the affinity graph is used to find bounaries between objects. If you train a network using `iterseg`, you can select the outputted network file to segment. Otherwise, if one is not selected, a network we have trained to detect platelets will be used. This might be appropriate for small objects with high anisotropy. 

#### DoG Blob Segmentation
The DoG blob segmentation uses a difference of Gaussian (DoG) filter to find blob shaped objects. The DoG filter is used to find object seeds, a foreground mask, and is fed to a watershed to label objects. This algorithm cannot be trained but can be configured with a configuration file. An example configuration file can be seen in the example folder in this repository. Please see the Segmentation_config.md file for more details. 

## Generating ground truth
We include two tools that are useful for generating ground truth: "save frames" and "ground truth from ROI". 

### Save frames

The first tool is "save frames" can be found at **plugins/iterseg/save_frames**. It enables you to save frames of interest from a  series of segmented images or timeseries. 

 ![save frames](https://github.com/AbigailMcGovern/iterseg/blob/main/docs/images/save_frames.png)

### Ground truth from ROI

The "ground truth from ROI" tool can be found at **plugins/iterseg/ground_truth_from_ROI**. This tool enables you to take a small portion of corrected data and place it into a new frame, which can be used for training. The new data can be tiled in the new frame to overrepresent the data in the training data set. At present, the ROI must be selected by adding a shapes layer (added using the icon circled in orange), then adding a rectangle (blue circle).

 ![make an ROI](https://github.com/AbigailMcGovern/iterseg/blob/main/docs/images/generate_ROI.png)

 The rectangle will be used to select a region of the xy-plane. This can be seen in 3D below. 

 ![2D ROI in 3D](https://github.com/AbigailMcGovern/iterseg/blob/main/docs/images/roi_before_3D.png)

 At present, the entire z stack above and below the rectangle will be used to generate ground truth. We aim to incorporate 3D bounding boxes in the future. If multiple ROIs are selected, multiple new image frames will be made, each with a single ROI. When you generate ground truth from the shapes layer, you are able to select the desired shapes layer, image layer, and labels layer. Additionally, you can choose how many times you want to tile the ROI and how much padding to leave between. Tiling will start at the top right and progress right before moving to the next row. You will also be able to choose the save name and the folder into which to save the data. 

 ![ground truth from ROI](https://github.com/AbigailMcGovern/iterseg/blob/main/docs/images/gt_from_ROI.png)

## Training a network
`iterseg` includes a widget for training a u-net for the u-net affinity watershed. The training widget can be found at **plugins/iterseg/train_from_viewer**. Before training, you will need to load the images and ground truth you want to train from. The images and ground truth should each be a series of 3D frames that are stacked into a layer (we suggest loading from a series of frames in a directory). Once loaded, you are able to select a layer as the ground truth and a layer as the image data. You can tell the program what the scale of the output frames will be (in the format (z, y, x)). You can select what type of center prediction to use (we suggest centredness), what type of prediction to use for the mask, and what extent of affinities you want to train (if n = 1, the network will predict only the direct boundaries between objects in each axis, if greater than 1 the network will still predict the direct boundaires but will also predict where there is a new object n steps away - can be used as collateral learning to enhance training). Affinities extent is developmental. Please submit an issue for any problems. 

 ![train from viewer](https://github.com/AbigailMcGovern/iterseg/blob/main/docs/images/train_from_viewer.png)

For the U-net training, we allow you to choose the learning rate for the [ADAM optimiser](https://arxiv.org/abs/1412.6980) used to train the network. You can also choose between binary cross entropy loss ([BCELoss](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html)) and Dice loss ([DICELoss](https://arxiv.org/abs/1707.03237v3)). We have found in our data that BCE loss works better. You can also choose how many chunks of data are produced from each frame (n each) and how many epochs you want to train for the training will be done in n_each * n_frames batches with a minibatch size of 1. 

In the future we hope to expand this training widget to enable training other types of networks. Please get involved if you feel you can help with this. 

## Assessing segmentations
`iterseg` includes widgets for assessing and comparing segmentations. If you want to assess segmentation quality, you will need to load a ground truth and a segmentation to assess. Once loaded, you can select the ground truth and segmentation (model segmentation) using the widget found in **plugins/iterseg/assess_segmentation**. You can select which metrics you want to assess. The metrics we enable are:
- **Variation of information (VI):** VI is a two part measure. It includes a measure of undersegmentation and oversegmentation. Undersegmentation is a measure of the amount of new information you get from looking at the ground truth if you have already seen the segmentation. It can be interpreted as the proportion of objects that are incorrectly merged. Oversegmentation is a measure of the amount of new information you get from looking at the segmentation if you have already seen the ground truth. It can be interpreted as the proportion of objects that are incorrectly split. For more info please see the [scikit-image documentation](https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.variation_of_information). 
- **Object count difference (OD):** Object count difference is simply the difference in number of objects between a ground truth and the assessed segmentation (card(ground truth) - card(segmentation)). 
- **Average precision (AP):** Average precision  Average precision is a combined measure of how accurate the model is at finding true positive (real) objects (we call this precision) and how many of ground truth real objects it found (this is called recall). The assessment of whether an object is TP, FP, and FN depends on the threashold of overlap between objects. Here we use the intersection of union (IoU), which is the proportion of overlap between the bounding boxes of ground truth and model segemented objects. AP is assessed using different IoU thresholds (from 0.35-0.95). The resultant data will be plotted as IoU by AP. 

  - Precision = TP / (TP + FP)Recall = TP / (TP + FN). 
  - Abbreviations: FN, false negative; TP, true positive; FP, false 


## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"iterseg" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[file an issue]: https://github.com/abigailmcgovern/iterseg/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
