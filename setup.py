import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name = 'annotrack',
    version = '0.0.1',
    author = 'Abigail McGovern',
    author_email = 'Abigail.McGovern1@monash.edu',
    description = 'Iteratively improving 3D cell segmentations using a unet-watershed approach',
    long_description = long_description,
    long_description_content_type = 'text/markdown', 
    license = 'BSD 2-Clause License',
    url = 'https://github.com/AbigailMcGovern/annotrack',
    project_urls = {
        'Bug Tracker' : 'https://github.com/AbigailMcGovern/annotrack/issues'
    },
    classifiers =
        ['Programming Language :: Python :: 3',
        'License :: OSI Approved :: BSD License', 
        'Operating System :: OS Independent', 
        'Intended Audience :: Science/Research', 
        'Topic :: Scientific/Engineering', 
        'Topic :: Scientific/Engineering :: Image Processing', ],
    packages = setuptools.find_packages(),
    python_requires = '>=3.6',
    install_requires =
        ['dask',
        'napari',
        'numpy',
        'torch',
        'zarr', 
        'zarpaint',
        'pytest', 
        'sphinx', 
        'pandas', 
        'ptitprince']
)