clustimage's documentation!
===========================

|python| |pypi| |docs| |stars| |LOC| |downloads_month| |downloads_total| |license| |forks| |open issues| |project status| |medium| |colab| |DOI| |repo-size| |donate|

.. tip::
	`Medium Blog: A step-by-step guide for clustering images <https://towardsdatascience.com/a-step-by-step-guide-for-clustering-images-4b45f9906128>`_


.. |fig1| image:: ../figs/schematic_overview.png

.. table:: 
   :align: center

   +----------+
   | |fig1|   |
   +----------+

-----------------------------------

The aim of ``clustimage`` is to detect natural groups or clusters of images.

Many computer vision tasks rely on (deep) neural networks, and aim to predict "what's on the image". However, not all tasks require supervised approaches or neural networks. With an unsupervised approach, we can aim to determine natural groups or clusters of images without being constrained to a fixed number of (learned) categories. In this blog, I will summarize the concepts of unsupervised clustering, followed by a hands-on tutorial on how to pre-process images, extract features (PCA, HOG), and group images with high similarity taking into account the goodness of the clustering. I will demonstrate the clustering of the MNIST dataset, the 101 objects dataset, the flower dataset, and finally the clustering of faces using the Olivetti dataset. All results are derived using the Python library clustimage.

``clustimage`` is a generic approach for unsupervised images clustering and overcomes the following challenges:

    * 1. Robustly groups similar images.
    * 2. Returns the unique images.
    * 3. Many plots for deeper exploration.
    * 4. Finds higly similar images for a given input image.


-----------------------------------

.. note::
	**Your ❤️ is important to keep maintaining this package.** You can `support <https://erdogant.github.io/clustimage/pages/html/Documentation.html>`_ in various ways, have a look at the `sponser page <https://erdogant.github.io/clustimage/pages/html/Documentation.html>`_.
	Report bugs, issues and feature extensions at `github <https://github.com/erdogant/clustimage/>`_ page.

	.. code-block:: console

	   pip install clustimage

-----------------------------------



Contents
========

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



.. |repo-size| image:: https://img.shields.io/github/repo-size/erdogant/clustimage
    :alt: repo-size
    :target: https://img.shields.io/github/repo-size/erdogant/clustimage

.. |stars| image:: https://img.shields.io/github/stars/erdogant/clustimage
    :alt: Stars
    :target: https://img.shields.io/github/stars/erdogant/clustimage

.. |python| image:: https://img.shields.io/pypi/pyversions/clustimage.svg
    :alt: |Python
    :target: https://erdogant.github.io/clustimage/

.. |pypi| image:: https://img.shields.io/pypi/v/clustimage.svg
    :alt: |Python Version
    :target: https://pypi.org/project/clustimage/

.. |docs| image:: https://img.shields.io/badge/Sphinx-Docs-blue.svg
    :alt: Sphinx documentation
    :target: https://erdogant.github.io/clustimage/

.. |LOC| image:: https://sloc.xyz/github/erdogant/clustimage/?category=code
    :alt: lines of code
    :target: https://github.com/erdogant/clustimage

.. |downloads_month| image:: https://static.pepy.tech/personalized-badge/clustimage?period=month&units=international_system&left_color=grey&right_color=brightgreen&left_text=PyPI%20downloads/month
    :alt: Downloads per month
    :target: https://pepy.tech/project/clustimage

.. |downloads_total| image:: https://static.pepy.tech/personalized-badge/clustimage?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=Downloads
    :alt: Downloads in total
    :target: https://pepy.tech/project/clustimage

.. |license| image:: https://img.shields.io/badge/license-MIT-green.svg
    :alt: License
    :target: https://github.com/erdogant/clustimage/blob/master/LICENSE

.. |forks| image:: https://img.shields.io/github/forks/erdogant/clustimage.svg
    :alt: Github Forks
    :target: https://github.com/erdogant/clustimage/network

.. |open issues| image:: https://img.shields.io/github/issues/erdogant/clustimage.svg
    :alt: Open Issues
    :target: https://github.com/erdogant/clustimage/issues

.. |project status| image:: http://www.repostatus.org/badges/latest/active.svg
    :alt: Project Status
    :target: http://www.repostatus.org/#active

.. |medium| image:: https://img.shields.io/badge/Medium-Blog-green.svg
    :alt: Medium Blog
    :target: https://erdogant.github.io/clustimage/pages/html/Documentation.html#medium-blog

.. |donate| image:: https://img.shields.io/badge/Support%20this%20project-grey.svg?logo=github%20sponsors
    :alt: donate
    :target: https://erdogant.github.io/clustimage/pages/html/Documentation.html#

.. |colab| image:: https://colab.research.google.com/assets/colab-badge.svg
    :alt: Colab example
    :target: https://erdogant.github.io/clustimage/pages/html/Documentation.html#colab-notebook

.. |DOI| image:: https://zenodo.org/badge/423822054.svg
    :alt: Cite
    :target: https://zenodo.org/badge/latestdoi/423822054


.. include:: add_bottom.add
