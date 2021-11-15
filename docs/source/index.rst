clustimage's documentation!
============================

The aim of ``clustimage`` is to detect natural groups or clusters of images.
Many computer vision tasks rely on (deep) neural networks, and aim to solve the problem of predicting "whats on the image".
However, not all tasks require supervised approaches, and it can be quit a breath to carefully group similar images in an unsupervised manner, or simply extract the unique images out of a huge set of images.
``clustimage`` is a generic approach for unsupervised images clustering and overcomes the following challenges: 
    * 1. Robustly groups similar images.
    * 2. Returns the unique images.
    * 3. Finds higly similar images for a given input image.


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
  :caption: Code Documentation
  
  Coding quality
  clustimage.clustimage



Quick install
-------------

.. code-block:: console

   pip install clustimage




Source code and issue tracker
------------------------------

Available on Github, `erdogant/clustimage <https://github.com/erdogant/clustimage/>`_.
Please report bugs, issues and feature extensions there.

Citing *clustimage*
--------------------
Here is an example BibTeX entry:

@misc{erdogant2019clustimage,
  title={clustimage},
  author={Erdogan Taskesen},
  year={2019},
  howpublished={\url{https://github.com/erdogant/clustimage}}}



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
