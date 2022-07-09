from clustimage.clustimage import Clustimage

from clustimage.clustimage import (
    import_example,
    wget,
    unzip,
    listdir,
    set_logger)

__author__ = 'Erdogan Tasksen'
__email__ = 'erdogant@gmail.com'
__version__ = '1.5.9'

# module level doc-string
__doc__ = """
clustimage
=====================================================================

Description
-----------
Python package clustimage is to detect natural groups or clusters of images.

The aim of ``clustimage`` is to detect natural groups or clusters of images. It works using a multi-step proces of carefully pre-processing the images, extracting the features, and evaluating the optimal number of clusters across the feature space.
The optimal number of clusters are determined using well known methods suchs as *silhouette, dbindex, and derivatives* in combination with clustering methods, such as *agglomerative, kmeans, dbscan and hdbscan*.
With ``clustimage`` we aim to determine the most robust clustering by efficiently searching across the parameter and evaluation the clusters.
Besides clustering of images, the ``clustimage`` model can also be used to find the most similar images for a new unseen sample.

Example
-------
>>> from clustimage import Clustimage
>>>
>>> # Init with default settings
>>> cl = Clustimage(method='pca')
>>>
>>> # load example with faces
>>> X = cl.import_example(data='mnist')
>>>
>>> # Cluster digits
>>> results = cl.fit_transform(X)
>>>
>>> # Cluster evaluation
>>> cl.clusteval.plot()
>>> cl.clusteval.scatter(cl.results['xycoord'])
>>> cl.pca.plot()
>>>
>>> # Unique
>>> cl.plot_unique(img_mean=False)
>>> cl.results_unique.keys()
>>>
>>> # Scatter
>>> cl.scatter(img_mean=False, zoom=3)
>>>
>>> # Plot clustered images
>>> cl.plot(labels=8)
>>>
>>> # Plot dendrogram
>>> cl.dendrogram()
>>>
>>> # Find images
>>> results_find = cl.find(X[0,:], k=None, alpha=0.05)
>>> cl.plot_find()
>>> cl.scatter()
>>>

References
----------
https://github.com/erdogant/clustimage

"""
