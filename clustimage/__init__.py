from clustimage.clustimage import Clustimage

from clustimage.clustimage import (
    import_example,
    wget,
    unzip,
    listdir,
    set_logger,
    )

__author__ = 'Erdogan Tasksen'
__email__ = 'erdogant@gmail.com'
__version__ = '1.2.2'

# module level doc-string
__doc__ = """
clustimage
=====================================================================

Description
-----------
Python package clustimage is for unsupervised clustering of images.
Clustering input images after following steps of pre-processing, feature-extracting, feature-embedding and cluster-evaluation.
Taking all these steps requires setting various input parameters. Not all input parameters can be changed across the different steps in clustimage.
Some parameters are choosen based on best practice, some parameters are optimized, while others are set as a constant.
The following 4 steps are taken:

Example
-------
>>> from clustimage import Clustimage
>>>
>>> # Init with default settings
>>> cl = Clustimage()
>>> # load example with flowers
>>> path_to_imgs = cl.import_example(data='flowers')
>>> # Detect cluster
>>> results = cl.fit_transform(path_to_imgs, min_clust=10)
>>>
>>> # Plot dendrogram
>>> cl.dendrogram()
>>> # Scatter
>>> cl.scatter(dotsize=50)
>>> # Plot clustered images
>>> cl.plot()
>>>
>>> # Make prediction
>>> results_find = cl.find(path_to_imgs[0:5], k=None, alpha=0.05)
>>> cl.plot_find()
>>> cl.scatter()
>>>

References
----------
https://github.com/erdogant/clustimage

"""
