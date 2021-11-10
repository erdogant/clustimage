.. _code_directive:

-------------------------------------

Preprocessing
''''''''''''''''

The functions that handle the preprocessing of images is concentrated into the function :func:`clustimage.clustimage.Clustimage.img_read_pipeline`.
This function reads the full path to the image that needs to be imported, and is subsequently colour-scaled (gray/rgb), scaled and resized.

The pre-processing has 4 steps and are exectued in this order.
    * 1. Import data.
    * 2. Scaling color pixels between [0-255]
    * 3. Conversion to gray-scale (user defined)
    * 4. Resizing (to reduce computation time)

.. code:: python

    # Import library
    import matplotlib.pyplot as plt
    import clustimage as cl
    # Load example dataset
    path_to_imgs = cl.import_example(data='flowers')
    # Run specific function to read preproces the images
    img = cl.img_read_pipeline(path_to_imgs[0], dim=(128,128))
    # Plot the flower after preprocessing    
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

   
   
Colorscale
''''''''''''

The *img_read* function :func:`clustimage.clustimage.img_read` reads the images and colour scales it based on the input parameter ``grayscale``. 
This function depends on functionalities from ``python-opencv`` and uses the ``cv2.COLOR_GRAY2RGB`` setting.


Scale
''''''''''''

The *scale* function :func:`clustimage.clustimage.img_scale` is only applicable for 2D-arrays (images).
Scaling data is an import pre-processing step to make sure all data is ranged between the minimum and maximum range.

The images are scaled between [0-255] by the following equation:

    Ximg * (255 / max(Ximg) )


Resize
''''''''''''

The resize function :func:`clustimage.clustimage.img_resize` resizes the images into 128x128 pixels (default) or an user-defined size.
The function depends on the functionality of ``python-opencv`` with the interpolation: ``interpolation=cv2.INTER_AREA``.


   