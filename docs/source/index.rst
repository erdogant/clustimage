clustimage's documentation!
============================

The aim of ``clustimage`` is to detect natural groups or clusters of images.

Many computer vision tasks rely on (deep) neural networks, and aim to predict "what's on the image". However, not all tasks require supervised approaches or neural networks. With an unsupervised approach, we can aim to determine natural groups or clusters of images without being constrained to a fixed number of (learned) categories. In this blog, I will summarize the concepts of unsupervised clustering, followed by a hands-on tutorial on how to pre-process images, extract features (PCA, HOG), and group images with high similarity taking into account the goodness of the clustering. I will demonstrate the clustering of the MNIST dataset, the 101 objects dataset, the flower dataset, and finally the clustering of faces using the Olivetti dataset. All results are derived using the Python library clustimage.

``clustimage`` is a generic approach for unsupervised images clustering and overcomes the following challenges:

    * 1. Robustly groups similar images.
    * 2. Returns the unique images.
    * 3. Many plots for deeper exploration.
    * 4. Finds higly similar images for a given input image.


.. tip::
	`For usage and more details, read the Medium blog: A step-by-step guide for clustering images <https://towardsdatascience.com/a-step-by-step-guide-for-clustering-images-4b45f9906128>`_


You contribution is important
==============================
If you ❤️ this project, **star** this repo at the `github page <https://github.com/erdogant/clustimage/>`_ and have a look at the `sponser page <https://erdogant.github.io/clustimage/pages/html/Documentation.html>`_!



Github
======
Please report bugs, issues and feature extensions at `github <https://github.com/erdogant/clustimage/>`_.



Content
=======

.. toctree::
   :maxdepth: 1
   :caption: Background

   Abstract


.. toctree::
   :maxdepth: 1
   :caption: Installation
   
   Installation


.. toctree::
  :maxdepth: 1
  :caption: Core functionalities

  core_functions

.. toctree::
  :maxdepth: 1
  :caption: Feature Extraction

  Feature Extraction


.. toctree::
  :maxdepth: 1
  :caption: Cluster Evaluation

  Cluster Evaluation

.. toctree::
  :maxdepth: 1
  :caption: Find/Predict

  Find


.. toctree::
  :maxdepth: 1
  :caption: Performance

  Performance

.. toctree::
  :maxdepth: 1
  :caption: Save and Load

  Save and Load


.. toctree::
  :maxdepth: 1
  :caption: Examples

  Examples


.. toctree::
  :maxdepth: 1
  :caption: Documentation

  Documentation
  Coding quality
  clustimage.clustimage





Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. raw:: html

	<hr>
	<center>
		<script async type="text/javascript" src="//cdn.carbonads.com/carbon.js?serve=CEADP27U&placement=erdogantgithubio" id="_carbonads_js"></script>
	</center>
	<hr>

