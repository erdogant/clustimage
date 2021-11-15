.. _code_directive:

-------------------------------------

Save and Load
''''''''''''''

Saving and loading models is desired as the learning proces of a model for ``clustimage`` can take up to hours.
In order to accomplish this, we created two functions: function :func:`clustimage.save` and function :func:`clustimage.load`
Below we illustrate how to save and load models.


Saving
----------------

Saving a learned model can be done using the function :func:`clustimage.save`:

.. code:: python

    import clustimage

    # Load example data
    X,y_true = clustimage.load_example()

    # Learn model
    model = clustimage.fit_transform(X, y_true, pos_label='bad')

    Save model
    status = clustimage.save(model, 'learned_model_v1')



Loading
----------------------

Loading a learned model can be done using the function :func:`clustimage.load`:

.. code:: python

    import clustimage

    # Load model
    model = clustimage.load(model, 'learned_model_v1')
