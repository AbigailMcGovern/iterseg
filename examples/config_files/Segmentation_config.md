# Configuring the segmentations


## Affinity U Net Watershed
The affinity-unet-watershed has only two parameters for the config file.
- `unet`: this argument specifies where to get the unet from. There are several options:
    - `None` will use the default network which was trained on platelets 
    - `"default"` will use the default network which was trained on platelets
    - `"labels layer"` will access the metadata in the reference layer
    - `"path/to/my/unet"` will use the unet at the specified file path (if the path exists)
- `affinites_extent`: an optional argument that will tell the algorithm what the affinities extent that the unet was trained to predict. This should be either `None` or an integer (`int`). The options include:
    - `None` will use the default value of 1
    - `1` will assume that only the first affinity was predicted. This is the default training setting for the unet. 
    - Any other `int` (integer) will specify the number of steps used to compute the affinity. 


## U Net Mask
The unet-mask has only one parameter for the configuration file. 
The affinity-unet-watershed has only two parameters for the config file.
- `unet`: this argument specifies where to get the unet from. There are several options:
    - `None` will use the default network which was trained on platelets. Only the mask channel will be used.
    - `"default"` will use the default network which was trained on platelets. Only the mask channel will be used.
    - `"labels layer"` will access the metadata in the reference layer. 
    - `"path/to/my/unet"` will use the unet at the specified file path (if the path exists). 

## Otsu Mask
The Otsu mask has only one parameter.
- `gaus_sigma`: 


## Blob Watershed
The blob watershed has five parameters. 
- `min_sigma`: Minimum sigma of a gaussian kernel to use for the laplacian of gaussian blob detection. The smaller this is, the smaller the smallest objects you detect will be.
- `max_sigma`: Maximum sigma to use for the 
- `num_sigma`: 
- `threshold`: 
- `gaus_sigma`: