Mnist dataset
#####################

In this example we will load the mnist dataset and cluster the images.

Load dataset
********************************

.. code:: python

	# Load library
	import matplotlib.pyplot as plt
	from clustimage import Clustimage
	# init
	cl = Clustimage()
	# Load example digit data
	X = cl.import_example(data='mnist')

	print(X)
	# Each row is an image that can be plotted after reshaping:
	plt.imshow(X[0,:].reshape(8,8), cmap='binary')
	# array([[ 0.,  0.,  5., ...,  0.,  0.,  0.],
	#        [ 0.,  0.,  0., ..., 10.,  0.,  0.],
	#        [ 0.,  0.,  0., ..., 16.,  9.,  0.],
	#        ...,
	#        [ 0.,  0.,  0., ...,  9.,  0.,  0.],
	#        [ 0.,  0.,  0., ...,  4.,  0.,  0.],
	#        [ 0.,  0.,  6., ...,  6.,  0.,  0.]])
	# 


Cluster the images
********************************

.. code:: python

	# Preprocessing and feature extraction
	results = cl.fit_transform(X)

	# Lets examine the results.
	print(results.keys())

	# ['feat', 'xycoord', 'pathnames', 'filenames', 'labels']
	# 
	# feat      : Extracted features
	# xycoord   : Coordinates of samples in the embedded space.
	# filenames : Name of the files
	# pathnames : Absolute location of the files
	# labels    : Cluster labels in the same order as the input


Detect unique images
********************************

.. code:: python

	# Get the unique images
	unique_samples = cl.unique()
	# 
	print(unique_samples.keys())
	# ['labels', 'idx', 'xycoord_center', 'pathnames']
	# 
	# Collect the unique images from the input
	X[unique_samples['idx'],:]


Cluster evaluation
********************************

.. code:: python

	# Plot the explained variance
	cl.pca.plot()
	# Make scatter plot of PC1 vs PC2
	cl.pca.scatter(legend=False, label=False)
	# Plot the evaluation of the number of clusters
	cl.clusteval.plot()

.. |figM7| image:: ../figs/digits_explained_var.png
.. |figM8| image:: ../figs/digits_clusters.png
.. table:: Explained variance and Sillhouette score
   :align: center

   +----------+----------+
   | |figM7|  | |figM8|  |
   +----------+----------+


.. code:: python

	# Make silhouette plot
	cl.clusteval.scatter(cl.results['xycoord'])


.. |figM9| image:: ../figs/digits_fig1.png
.. table:: Sillhouette analysis results in 9 clusters.
   :align: center

   +----------+
   | |figM9|  |
   +----------+



Scatter plot
********************************

The scatterplot that is coloured on the clusterlabels. The clusterlabels should match the unique labels.
Cluster 1 contains digit 4, and  Cluster 5 contains digit 2, etc.

.. code:: python

	# Make scatterplot
	cl.scatter(zoom=None)

	# Plot the image that is in the center of the cluster
	cl.scatter(zoom=4)


.. |figM1| image:: ../figs/digits_fig2_tsne.png
.. |figM2| image:: ../figs/digits_fig21_tsne.png
.. table:: Left: Scatter plot with cluster labels of all samples. Right: scatter plot with unique image in center.
   :align: center

   +----------+----------+
   | |figM1|  | |figM2|  |
   +----------+----------+

High resolution images where all mnist samples are shown.

.. code:: python

	cl.scatter(zoom=8, plt_all=True, figsize=(150,100))


.. |figM3| image:: ../figs/scatter_mnist_all.png
.. table:: Left: Scatter plot with cluster labels of all samples. Right: scatter plot with unique image in center.
   :align: center

   +----------+
   | |figM3|  |
   +----------+


Plot images detected in a cluster
************************************************

.. code:: python

	# Plot all images per cluster
	cl.plot(cmap='binary')

	# Plot the images in a specific cluster
	cl.plot(cmap='binary', labels=[1,5])


.. |figM4| image:: ../figs/digits_cluster1.png
.. |figM5| image:: ../figs/digits_cluster5.png
.. table:: Images that are detected in a particular cluster.
   :align: center

   +----------+----------+
   | |figM4|  | |figM5|  |
   +----------+----------+


Dendrogram
************************************************

.. code:: python

	# The dendrogram is based on the high-dimensional feature space.
	cl.dendrogram()


.. |figM6| image:: ../figs/digits_dendrogram.png
.. table:: Dendrogram of the mnist dataset
   :align: center

   +----------+
   | |figM6|  |
   +----------+





Caltech101 dataset
#####################

The documentation and docstrings readily contains various examples but lets make another one with many samples.
In this example, the **Caltech101** dataset will be clustered!
The pictures of objects belonging to 101 categories. About 40 to 800 images per category. Most categories have about 50 images. The size of each image is roughly 300 x 200 pixels.
Download the dataset over here: http://www.vision.caltech.edu/Image_Datasets/Caltech101/#Download

Cluster the images
********************************

.. code:: python

    from clustimage import Clustimage

    # init
    cl = Clustimage(method='pca', params_pca={'n_components':250})
    
    # Collect samples
    # Preprocessing, feature extraction and cluster evaluation
    results = cl.fit_transform('C://101_ObjectCategories//', min_clust=30, max_clust=60)
    
    # Try some other clustering (evaluation) approaches
    # cl.cluster(evaluate='silhouette', min_clust=30, max_clust=60)
    

Cluster evaluation
********************************

With ``clustimage`` we extracted the features that explained 89% of the variance. The optimal number of clusters of 49 (right figure).


.. code:: python

    # Evaluate the number of clusters.
    cl.clusteval.plot()
    cl.clusteval.scatter(cl.results['xycoord'])


.. |figE1| image:: ../figs/101_explainedvar.png
.. |figE2| image:: ../figs/101_optimalclusters.png
.. |figE3| image:: ../figs/101_silhouette_plot.png
.. table:: Left: Percentage explained variance. Right: Optimal number of clusters.
   :align: center

   +----------+----------+
   | |figE1|  | |figE2|  |
   +----------+----------+


Silhouette Plot
********************************

.. code:: python

    # Plot one of the clusters
    cl.plot(labels=40)
    
    # Plotting
    cl.dendrogram()


.. |figE3| image:: ../figs/101_silhouette_plot.png
.. table:: Silhouette plot
   :align: center

   +----------+
   | |figE3|  |
   +----------+


Average image per cluster
********************************

For each of the detected clusters, we can collect the images and plot the image in the center (left figure), or we can average all images to a single image (right figure).

.. code:: python

    # Plot unique images. 
    cl.plot_unique()
    cl.plot_unique(img_mean=False)

.. |figE8| image:: ../figs/unique_mean_101.png
.. |figE9| image:: ../figs/unique_mean_101.png

.. table:: Left: Unique images gathered from the center of the cluster. Right: Averaged image of the cluster.
   :align: center

   +----------+----------+
   | |figE8|  | |figE9|  |
   +----------+----------+


Scatter plot
********************************

A scatter plot demonstrates the samples with its cluster labels (colors), and the average images per cluster.

.. code:: python

    # Scatter
    cl.scatter(dotsize=10, img_mean=False, zoom=None)
    cl.scatter(dotsize=10, img_mean=False)
    cl.scatter(dotsize=10)


.. |figE4| image:: ../figs/101_tsne_no_mean.png
.. |figE5| image:: ../figs/101_tsne.png
.. table:: Left: Unique images gathered from the center of the cluster. Right: Averaged image of the cluster.
   :align: center

   +----------+----------+
   | |figE4|  | |figE5|  |
   +----------+----------+




Plot images detected in a cluster
************************************************

.. |figE6| image:: ../figs/101_cluster40.png
.. |figE7| image:: ../figs/101_cluster.png
.. table:: Two examples of the clusters that are detected.
   :align: center

   +----------+----------+
   | |figE6|  | |figE7|  |
   +----------+----------+




.. raw:: html

	<hr>
	<center>
		<script async type="text/javascript" src="//cdn.carbonads.com/carbon.js?serve=CEADP27U&placement=erdogantgithubio" id="_carbonads_js"></script>
	</center>
	<hr>
