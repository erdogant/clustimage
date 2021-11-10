.. _code_directive:

-------------------------------------

PCA
''''''''''

Principal component analysis (PCA) is the process of computing the principal components.
In ``clustimage`` the first 50 (default) components are selected and the rest is ignoring.
The use of PC's for the clustering is usefull in applications with among others faces, where so called eigenfaces are computed.
The eigenface is a low-dimensional representation of face images. It is shown that principal component analysis could be used on a collection of face images to form a set of basis features.

HOG
''''''''''

Histogram of Oriented Gradients (HOG), is a feature descriptor that is often used to extract features from image data. 
In general, it is a simplified representation of the image that contains only the most important information about the image.
The HOG feature descriptor counts the occurrences of gradient orientation in localized portions of an image. It is widely used in computer vision tasks for object detection.

 * The HOG descriptor focuses on the structure or the shape of an object. Note that this is different then **edge features** that we can extract for images because in case of HOG features, both edge and direction are extracted.
 * The complete image is broken down into smaller regions (localized portions) and for each region, the gradients and orientation are calculated.
 * Finally the HOG would generate a Histogram for each of these regions separately. The histograms are created using the gradients and orientations of the pixel values, hence the name **Histogram of Oriented Gradients**

Not all applications are usefull when using HOG features as it "only" provides the outline of the image.
For example, if the use-case is to group faces or cars, HOG-features can do a great job but a deeper similarity of persons or car-types may not as the details will be losed.


.. code:: python

    import matplotlib.pyplot as plt
    from clustimage import Clustimage
    # init
    cl = Clustimage(method='hog')
    # Load example data
    path_to_imgs = cl.import_example(data='flowers')
    # Set dim to (128,128)
    dim = (128,128)
    # Read image according the preprocessing steps
    img = cl.img_read_pipeline(path_to_imgs[0], dim=dim)
    # Extract HOG features.
    img_hog = cl.extract_hog(img, pixels_per_cell=(16, 16))
    
    plt.figure();
    fig,axs=plt.subplots(1,2)
    axs[0].imshow(img.reshape([dim[0],dim[1],3]))
    axs[0].axis('off')
    axs[0].set_title('Preprocessed image', fontsize=10)
    axs[1].imshow(img_hog.reshape(dim), cmap='binary')
    axs[1].axis('off')
    axs[1].set_title('HOG', fontsize=10)


.. |figF1| image:: ../figs/hog_example.png

.. table:: HOG example containing 8x8 vectors
   :align: center

   +----------+
   | |figF1|  |
   +----------+

Here it can be clearly seen that the HOG image is a matrix of 8x8 vectors that is derived by because of the input image (128,128) devided by the pixels per cell (16,16). Thus 128/16=8 rows and columns in this case.
If an increase of HOG features is desired, you can either increasing the image dimensions (eg 256,256) or decrease the pixels per cell (eg 8,8).
   

.. code:: python

    # Extract HOG features.
    img_hog = cl.extract_hog(img, pixels_per_cell=(8, 8))


.. |figF2| image:: ../figs/hog_example88.png

.. table:: HOG example containing 16x16 vectors
   :align: center

   +----------+
   | |figF2|  |
   +----------+

   
tSNE
''''''''''

t-distributed stochastic neighbor embedding can be used to transform the samples into a 2D space.

