.. _code_directive:

-------------------------------------

The documentation and docstrings readily contains various examples but lets make another one with many samples.

Caltech101 dataset
''''''''''''''''''''

Lets use ``clustimage`` on a the **Caltech101** dataset to clusters the images.
The pictures of objects belonging to 101 categories. About 40 to 800 images per category. Most categories have about 50 images. The size of each image is roughly 300 x 200 pixels.
Download the dataset over here: http://www.vision.caltech.edu/Image_Datasets/Caltech101/#Download

.. code:: python

    from clustimage import Clustimage
    # init
    cl = Clustimage(method='pca', params_pca={'n_components':250})
    # Collect samples
    # Preprocessing, feature extraction and cluster evaluation
    results = cl.fit_transform('C://101_ObjectCategories//', min_clust=30, max_clust=60)
    # Try some other clustering (evaluation) approaches
    # cl.cluster(evaluate='silhouette', min_clust=30, max_clust=60)
    # Evaluate the number of clusters.
    cl.clusteval.plot()
    cl.clusteval.scatter(cl.results['xycoord'])
    # Plot unique images. When comparing the unique images that are centered in the cluster vs. the average cluster imge, some clusters appear very strong.
    cl.plot_unique()
    cl.plot_unique(img_mean=False)
    # Scatter
    cl.scatter(dotsize=10, img_mean=False, zoom=None)
    cl.scatter(dotsize=10, img_mean=False)
    cl.scatter(dotsize=10)
    # Plot one of the clusters
    cl.plot(labels=40)
    # Plotting
    cl.dendrogram()




With ``clustimage`` we could easily extract the features that explains 89% of the variance and detected an optimal number of clusters of 49.

.. |figE1| image:: ../figs/101_explainedvar.png
.. |figE2| image:: ../figs/101_optimalclusters.png
.. |figE3| image:: ../figs/101_silhouette_plot.png

.. table:: Left: Percentage explained variance. Right: Optimal number of clusters.
   :align: center

   +----------+----------+
   | |figE1|  | |figE2|  |
   +----------+----------+



.. |figE8| image:: ../figs/unique_mean_101.png
.. |figE9| image:: ../figs/unique_mean_101.png

.. table:: Left: Unique images gathered from the center of the cluster. Right: Averaged image of the cluster.
   :align: center

   +----------+----------+
   | |figE8|  | |figE9|  |
   +----------+----------+



.. |figE3| image:: ../figs/101_silhouette_plot.png

.. table:: Silhouette plot
   :align: center

   +----------+
   | |figE3|  |
   +----------+

.. |figE4| image:: ../figs/101_tsne_no_mean.png
.. |figE5| image:: ../figs/101_tsne.png


.. table:: Left: Unique images gathered from the center of the cluster. Right: Averaged image of the cluster.
   :align: center

   +----------+----------+
   | |figE4|  | |figE5|  |
   +----------+----------+



.. |figE6| image:: ../figs/101_cluster40.png
.. |figE7| image:: ../figs/101_cluster.png

.. table:: Two examples of the clusters that are detected.
   :align: center

   +----------+----------+
   | |figE6|  | |figE7|  |
   +----------+----------+




