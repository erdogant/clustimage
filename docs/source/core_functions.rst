.. _code_directive:

-------------------------------------

This section describes the core functionalities of ``clustimage``.
Many of the functionalities are written in a generic manner which allows to be used in various applications.

Core functionalities
''''''''''''''''''''''
The are 5 core functionalities of ``clustimage`` that allows to preprocess the input images, robustly determines the optimal number of clusters, and then optimize the clusters if desired.

    * fit_transform
    * detect_faces
    * cluster
    * find
    * unique
    
Fit and transform
^^^^^^^^^^^^^^^^^^^^
The *fit_transform* function allows to detect natural groups or clusters of images. It works using a multi-step proces of pre-processing, extracting the features, and evaluating the optimal number of clusters across the feature space.
The optimal number of clusters are determined using well known methods such as *silhouette, dbindex, and derivatives* in combination with clustering methods, such as *agglomerative, kmeans, dbscan and hdbscan*.
Based on the clustering results, the unique images are also gathered.

Examples can be found here: :func:`clustimage.clustimage.Clustimage.fit_transform`


detect_faces
^^^^^^^^^^^^^^
To cluster faces on images, we first need to detect, and extract the faces from the images.
The *detect_faces* function does this task.
Faces and eyes are detected using ``haarcascade_frontalface_default.xml`` and ``haarcascade_eye.xml`` in ``python-opencv``.

Examples can be found here: :func:`clustimage.clustimage.Clustimage.detect_faces`


cluster
^^^^^^^^^
The *cluster* function is build on ``clusteval``, which is a python package that provides various evalution methods for unsupervised cluster validation.
The optimal number of clusters are determined using well known methods such as *silhouette, dbindex, and derivatives* in combination with clustering methods, such as *agglomerative, kmeans, dbscan and hdbscan*.
This function can be run after the ``fit_transform`` function to solely optimize the clustering results or try-out different evaluation approaches without repeately performing all the steps of preprocessing.
Besides changing evaluation methods and metrics, it is also possible to cluster on the low-embedded feature space. This can be done setting the parameter ``cluster_space='low'``.

Examples can be found here: :func:`clustimage.clustimage.Clustimage.cluster`

find
^^^^^^^
The *find* function is to detect images that are similar to that of the input image.
Finding images can be used in two manners:

    * Based on the k-nearest neighbour 
    * Based on significance after probability density fitting 

In both cases, first the adjacency matrix is computed using the distance metric (default Euclidean).
In case of the K-nearest neighbour approach, the k nearest neighbours are determined.
In case of significance, the adjacency matrix is used to to estimate the loc/scale/arg parameters for various theoretical distribution.
The tested disributions are *['norm', 'expon', 'uniform', 'gamma', 't']*. The fitted distribution describes the similarity-distribution of samples.
For each new image, the probability is computed, and returns the images that are <= *alpha* in the lower bound of the distribution.
If both k and alpha is specified, the union of detected samples is taken.

Examples can be found here: :func:`clustimage.clustimage.Clustimage.find`

unique
^^^^^^^^^^
The unique images are detected by first computing the center of the cluster, and then taking the image closest to the center.

:func:`clustimage.clustimage.Clustimage.unique`


Preprocessing
''''''''''''''''

The preprocessing step is the function :func:`clustimage.clustimage.Clustimage.imread`, and contains 3 functions to handle the import, scaling and resizing of images.
This function requires the full path to the image for which the first step is reading the images and colour scaling it based on the input parameter ``grayscale``.
If ``grayscale`` is set to *True*, the ``cv2.COLOR_GRAY2RGB`` setting from ``python-opencv`` is used.

The pre-processing has 4 steps and are exectued in this order.

    * 1. Import data.
    * 2. Conversion to gray-scale (user defined)
    * 3. Scaling color pixels between [0-255]
    * 4. Resizing

.. code:: python

    # Import libraries
    from clustimage import Clustimage
    import matplotlib.pyplot as plt

    # Init
    cl = Clustimage()
    # Load example dataset
    pathnames = cl.import_example(data='flowers')
    # Preprocessing of the first image
    img = cl.imread(pathnames[0], dim=(128,128))

    # Plot
    plt.figure()
    plt.imshow(img.reshape(128,128,3))
    plt.axis('off')


.. |figP1| image:: ../figs/flower_original.png
.. |figP2| image:: ../figs/flower_example1.png

.. table:: Left is orignal input figure and right is after preprocessing
   :align: center

   +----------+----------+
   | |figP1|  | |figP2|  | 
   +----------+----------+



imscale
^^^^^^^^

The *imscale* function :func:`clustimage.clustimage.Clustimage.imscale` is only applicable for 2D-arrays (images).
Scaling data is an import pre-processing step to make sure all data is ranged between the minimum and maximum range.

The images are scaled between [0-255] by the following equation:

    Ximg * (255 / max(Ximg) )


imresize
^^^^^^^^^

The *imresize* function :func:`clustimage.clustimage.imresize` resizes the images into 128x128 pixels (default) or to an user-defined size.
The function depends on the functionality of ``python-opencv`` with the interpolation: ``interpolation=cv2.INTER_AREA``.




Generic functionalities
''''''''''''''''''''''''
``clustimage`` contains various generic functionalities that are internally used but may be usefull too in other applications.

wget
^^^^^^^^^
Download files from the internet and store on disk.
Examples can be found here: :func:`clustimage.clustimage.wget`

.. code:: python

    # Import library
    import clustimage as cl
    # Download
    images = cl.wget('https://erdogant.github.io/datasets/flower_images.zip', 'c://temp//flower_images.zip')


unzip
^^^^^^^^^
Unzip files into a destination directory.
Examples can be found here: :func:`clustimage.clustimage.unzip`

.. code:: python

    # Import library
    import clustimage as cl
    # Unzip to path
    dirpath = cl.unzip('c://temp//flower_images.zip')


listdir
^^^^^^^^^
Recusively list the files in the directory.
Examples can be found here: :func:`clustimage.clustimage.listdir`

.. code:: python

    # Import library
    import clustimage as cl
    # Unzip to path
    dirpath = 'c://temp//flower_images'
    pathnames = cl.listdir(dirpath, ext=['png'])


set_logger
^^^^^^^^^^^^
Change status of the logger.
Examples can be found here: :func:`clustimage.clustimage.set_logger`

.. code:: python

    # Change to verbosity message of warnings and higher
    set_logger(verbose=30)


extract_hog
^^^^^^^^^^^^
Extracting hog features.
Examples can be found here: :func:`clustimage.clustimage.Clustimage.extract_hog`

.. code:: python

    import matplotlib.pyplot as plt
    from clustimage import Clustimage
    
    # Init
    cl = Clustimage(method='hog')
    
    # Load example data
    pathnames = cl.import_example(data='flowers')
    # Read image according the preprocessing steps
    img = cl.imread(pathnames[0], dim=(128,128))
    
    # Extract HOG features
    img_hog = cl.extract_hog(img)
    
    plt.figure();
    fig,axs=plt.subplots(1,2)
    axs[0].imshow(img.reshape(128,128,3))
    axs[0].axis('off')
    axs[0].set_title('Preprocessed image', fontsize=10)
    axs[1].imshow(img_hog.reshape(128,128), cmap='binary')
    axs[1].axis('off')
    axs[1].set_title('HOG', fontsize=10)


.. |figC1| image:: ../figs/hog_example.png

.. table:: Left is orignal input figure and right the hog features
   :align: center

   +----------+
   | |figC1|  | 
   +----------+

