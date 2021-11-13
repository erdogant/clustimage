.. _code_directive:

-------------------------------------

Installation
''''''''''''

If desired, install ``clustimage`` from an isolated Python environment using conda:

.. code-block:: python

    conda create -n env_clustimage python=3.8
    conda activate env_clustimage

Install via ``pip``:

.. code:: bash

    # Intstall the library
    pip install clustimge


Uninstalling
''''''''''''

If you want to remove your ``clustimage`` installation with your environment:

.. code-block:: console

   # List all the active environments. clustimage should be listed.
   conda env list

   # Remove the clustimage environment
   conda env remove --name clustimage

   # List all the active environments. clustimage should be absent.
   conda env list


Quickstart
''''''''''

A quick example how to learn a model on a given dataset.

.. code:: python

    # Import library
    from clustimage import Clustimage

    # init with default parameters
    cl = Clustimage()

    # load example with flowers
    path_to_imgs = cl.import_example(data='flowers')

    # Run the model to find the optimal clusters
    results = cl.fit_transform(path_to_imgs, min_clust=10)

    # Plot dendrogram
    cl.dendrogram()
    # Scatter
    cl.scatter(dotsize=50)
    # Plot clustered images
    cl.plot()

    # Predict
    results_find = cl.find(path_to_imgs[0:5], k=None, alpha=0.05)
    cl.plot_find()
    cl.scatter()
