.. _code_directive:

-------------------------------------

Hyperparameter tuning
''''''''''''''''''''''''''''

By gridsearch
 
In ``urldetect`` we incorporated hyperparameter optimization using a gridseach :func:`urldetect._gridsearch`. The goal is to evaluate the value of the combination of parameters in the learning process.
The use of gridsearch is set True as default by a boolean value ``gridsearch=True`` in the function :func:`urldetect.fit_transform` or :func:`urldetect.fit`.
You may want to set this value at ``gridsearch=False`` if the number of samples is very low which would lead in a poorly trained model.


Silhouette
'''''''''''

To measure the goodness of fit we use various evaluation metrics to check the classification model’s performance.
The performance scores can be derived in ``urldetect`` using in the function :func:`urldetect.plot`.

The performance of the model can deviate based on the threshold being used but the theshold this will not affect the learning process :func:`urldetect.fit_transform`.
After learning a model, and predicting new samples with it, each sample will get a probability belowing to the class. In case of our two-class approach the simple rule account: **P(class malicous) = 1-P(class normal)**
The threshold is used on the probabilities to devide samples into the malicous or normal class.




dbindex
'''''''''

A confusion matrix is a table that is often used to describe the performance of a classification model (or “classifier”) 
on a set of test data for which the true values are known. It allows the visualization of the performance of an algorithm.

Cohen's kappa coefficient is a statistic that is used to measure inter-rated reliability for qualitative (categorical) items.

.. code:: python

    scoring = make_scorer(cohen_kappa_score, greater_is_better=True)


dbscan
''''''''

The probability graph plots the probabilities of the samples being classified.


hdbscan
'''''''''

The classification performance can be derived using the function :func:`urldetect.plot`. 
Results for the malicous URLs, using a 5-fold crossvalidation with gridsearch is as follows:

.. _Figure_1:

.. figure:: ../figs/Figure_1.png
    :scale: 80%

.. _Figure_2:

.. figure:: ../figs/Figure_2.png
    :scale: 80%

.. _Figure_3:

.. figure:: ../figs/Figure_3.png
    :scale: 80%

