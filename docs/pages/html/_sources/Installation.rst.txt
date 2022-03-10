Installation
################

Create environment
**********************

If desired, install ``clustimage`` from an isolated Python environment using conda:

.. code-block:: python

    conda create -n env_clustimage python=3.8
    conda activate env_clustimage


Pypi
**********************

.. code-block:: console

    # Install from Pypi:
    pip install clustimage

    # Force update to latest version
    pip install -U clustimage


Github source
************************************

.. code-block:: console

    # Install directly from github
    pip install git+https://github.com/erdogant/clustimage


Uninstalling
################

Remove environment
**********************

.. code-block:: console

   # List all the active environments. clustimage should be listed.
   conda env list

   # Remove the clustimage environment
   conda env remove --name clustimage

   # List all the active environments. clustimage should be absent.
   conda env list


Remove installation
**********************

Note that the removal of the environment will also remove the ``clustimage`` installation.

.. code-block:: console

    # Install from Pypi:
    pip uninstall clustimage



.. raw:: html

	<hr>
	<center>
		<script async type="text/javascript" src="//cdn.carbonads.com/carbon.js?serve=CEADP27U&placement=erdogantgithubio" id="_carbonads_js"></script>
	</center>
	<hr>



Quickstart
**********************

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

    # Cluster evaluation plot
    cl.clustimage.plot()
    
    # Unique images
    cl.results_unique.keys()
    cl.plot_unique(img_mean=False)

    # Scatter
    cl.scatter(dotsize=50, img_mean=False)

    # Plot clustered images
    cl.plot(labels=0)

    # Plot dendrogram
    cl.dendrogram()

    # Predict
    results_find = cl.find(path_to_imgs[0:5], k=None, alpha=0.05)
    cl.plot_find()
    cl.scatter()


.. raw:: html

	<hr>
	<center>
		<script async type="text/javascript" src="//cdn.carbonads.com/carbon.js?serve=CEADP27U&placement=erdogantgithubio" id="_carbonads_js"></script>
	</center>
	<hr>
