.. _code_directive:

-------------------------------------

Quickstart
''''''''''

A quick example how to learn a model on a given dataset.


.. code:: python

    # Import library
    import clustimage

    # Retrieve URLs of malicous and normal urls:
    X, y = clustimage.load_example()

    # Learn model on the data
    model = clustimage.fit_transform(X, y, pos_label='bad')

    # Plot the model performance
    results = clustimage.plot(model)


Installation
''''''''''''

Create environment
------------------


If desired, install ``clustimage`` from an isolated Python environment using conda:

.. code-block:: python

    conda create -n env_clustimage python=3.6
    conda activate env_clustimage


Install via ``pip``:

.. code-block:: console

    # The installation from pypi is disabled:
    pip install clustimage

    # Install directly from github
    pip install git+https://github.com/erdogant/clustimage


Uninstalling
''''''''''''

If you want to remove your ``clustimage`` installation with your environment, it can be as following:

.. code-block:: console

   # List all the active environments. clustimage should be listed.
   conda env list

   # Remove the clustimage environment
   conda env remove --name clustimage

   # List all the active environments. clustimage should be absent.
   conda env list
