Saving
##########

Saving and loading models can be used to restore previous results.
In order to accomplish this, we created two functions: function :func:`clustimage.clustimage.Clustimage.save`. and function :func:`clustimage.clustimage.Clustimage.load`.
Below we illustrate how to save and load models.

Saving the model with the results:

.. code:: python

    from clustimage import Clustimage

    # Initialize
    cl = Clustimage(method='hog')
    # Load example data

    X = cl.import_example(data='mnist')
    # Preprocessing, feature extraction and cluster evaluation
    results = cl.fit_transform(X)

    # Load model
    cl.save('clustimage_model', overwrite=True)


Loading
##########

Loading a learned model can be done using the function :func:`clustimage.load`:

.. code:: python

    from clustimage import Clustimage

    # Load model
    cl.load('clustimage_model')


.. raw:: html

	<hr>
	<center>
		<script async type="text/javascript" src="//cdn.carbonads.com/carbon.js?serve=CEADP27U&placement=erdogantgithubio" id="_carbonads_js"></script>
	</center>
	<hr>
