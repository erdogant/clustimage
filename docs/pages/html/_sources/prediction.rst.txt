.. _code_directive:

-------------------------------------

Prediction
''''''''''''''''''''''''''''

The performance of the model can deviate based on the threshold being used but the theshold this will not affect the learning process :func:`urldetect.fit_transform`.
After learning a model, and predicting new samples with it, each sample will get a probability belowing to the class. In case of our two-class approach the simple rule account: **P(class malicous) = 1-P(class normal)**
The threshold is used on the probabilities to devide samples into the malicous or normal class.


For the detection of the optimal number of clusters, the python library ``clusteval`` is utilized to evaluate the **goodness** of clusters.
The clustering approaches can be set to *agglomerative*, *kmeans*, *dbscan* and *hdbscan*, for which the ``clusteval`` library then searches across the space of clusters and method-parameters to determine the optimal number of clusters given the input dataset.

**Cluster evaluation can be performed based on:**

    * Silhouette scores
    * DBindex
    * Derivative method

Lets load the **digits** dataset and see how the different methods detects the optimal number of clusters.

.. code:: python

    from clustimage import Clustimage
    # init
    cl = Clustimage(method='pca', embedding='tsne', cluster_space='high', grayscale=True, store_to_disk=True)
    # Example data
    X = cl.import_example(data='mnist')



.. code:: python

    # Feature extraction and cluster evaluation
    results = cl.fit_transform(X, evaluate='silhouette', cluster='agglomerative')

    # Cluster differently using directly the extracted features.
    # results = cl.cluster(, cluster='agglomerative', evaluate='silhouette', cluster_space='low')

    # Scatter
    cl.scatter()
    # Dendrogram with cluster evalution
    cl.dendrogram()


.. |figCE0| image:: ../figs/digits_clusters.png
.. |figCE6| image:: ../figs/silhouette_tsne.png

.. table:: Left: The number of clusters vs silhouette scores. Right: tSNE plot coloured on the cluster-labels.
   :align: center

   +----------+----------+
   | |figCE0| | |figCE6| |
   +----------+----------+
