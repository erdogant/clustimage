.. _code_directive:

-------------------------------------

Abstract
''''''''

Background

Image recognition is a computer vision task for identifying and verifying objects/persons on a photograph.
We can seperate the image recognition task into the two broad tasks, namely the supervised and unsupervised task.
In case of the supervised task, we have to classify an image into a fixed number of learned categories.
In case of the unsupervised task, we do not depend on the fact that training data is required but we can interpret the input data and find natural groups or clusters.

Aim

The aim of ``clustimage`` is to detect natural groups or clusters of images. It works using a multi-step proces of carefully pre-processing the images, extracting the features, and evaluating the optimal number of clustering across the the high-dimensional feature space.
The optimal number of clusters can be determined using the three well known methods; silhouette, dbindex, and derivatives in combination with four clustering methods; agglomerative, kmeans, dbscan and hdbscan.
With ``clustimage`` we aim to determine the most robust clustering by efficiently searching across the parameter and evaluation the clusters.
Besides clustering of images, the final optimized model can also be used to predict the most similar image for a new unseen samples.

Results

``clustimage`` is Python package for unsupervised clustering of images after carefully pre-processing the images, extracting the features, and evaluating the optimal number of clustering in the high-dimensional feature space.
``clustimage`` does not depend on complex pre-trained neural networks with many package-dependencies but simply extract features using Principal component analysis (PCA), and/or Histogram of Oriented Gradients (HOG).
The input for the model can be a NxM array for which each rows is a flattened rgb/gray image, or it can be a target directory or the full path to a list of images.
    

Schematic overview
'''''''''''''''''''

The schematic overview of our approach is as following:

.. _schematic_overview:

.. figure:: ../figs/schematic_overview.png
