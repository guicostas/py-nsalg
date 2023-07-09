
Sklearn-pandas
==============

.. image:: https://circleci.com/gh/scikit-learn-contrib/sklearn-pandas.svg?style=svg
    :target: https://circleci.com/gh/scikit-learn-contrib/sklearn-pandas
.. image:: https://img.shields.io/pypi/v/sklearn-pandas.svg
   :target: https://pypi.python.org/pypi/sklearn-pandas/
.. image:: https://anaconda.org/conda-forge/sklearn-pandas/badges/version.svg
   :target: https://anaconda.org/conda-forge/sklearn-pandas/

.. highlight:: python

This module provides a bridge between `Scikit-Learn <http://scikit-learn.org/stable>`__'s machine learning methods and `pandas <https://pandas.pydata.org>`__-style Data Frames.
In particular, it provides a way to map ``DataFrame`` columns to transformations, which are later recombined into features.

Installation
------------

You can install ``py-nsalg`` with ``pip``::

    # pip install py-nsalg

or conda-forge::

    # conda install -c conda-forge py-nsalg

Tests
-----

The examples in this file double as basic sanity tests. To run them, use ``doctest``, which is included with python::

    # python -m doctest README.rst


Usage
-----


Import
******

Import what you need from the ``py-nsalg`` package. you may want to use pandas and perform some transformations before applying NSA at your data.


Load some Data
**************


Feel free to bring your data from most convenient file type, such as CSV, JSON, AIFF (WEKA), some scikit-learn databases, among other types. Pay attention: you should convert your data to array from numpy, resulting in a N x F array, where N is the number of samples and F is the number of features considered. Since the algorithm accepts only real numbers at the moment, perform proper transformations at your data by using OneHotEncoding for example.


Generating the model
*********************

In order to generate the set of detectors, you should separate some samples as both training and test data, since NSA is a supervised method of anomaly detection. You should also filter the training as only normal class data is considered. If you use anomalous data in your training, your test may present some false positive rate.

Note that the tracker generation is performed randomly, i.e. the algorithm presents different results for every execution for your database, regardless of parameters used. Some false negative rate can be presented as the method has some drawbacks.

Check the file `testing.py` for an example of algorithm usage.


Algorithms implemented
***********************


Two variations of NSA are presented here:

* Classical Negative Selection, with constant radius size and naive self data checking
* V-Detector Method, with variable radius size and coverage analysis

The function `evalNSA` performs the execution of both training and test steps. For bidimensional data, a graph from matplotlib is presented displaying training data with trackers generated, as well as another graph with the detection results.


Changelog
---------

0.5 (2023-07-09)
********************

* Initial release
* Provided two approaches of NSA-based algorithms for testing phase and one matching function for testing phase. Euclidean distance is applied to compare trackers and samples in the shape space of features. Results are displayed in a matplotlib graph if verbose is set to 1 and for bidimensional datasets. This algorithm is applied only for real-valued problems for now.


What's next?
---------
* Refactoring and compatibility with *scikit-learn*
* Compatibility with *pandas*
* Compatibility with active learning and semi-supervised learning methods
* Appropriate normalization methods based on scaling, standardization and others
* Evaluation of other types besides real numbers
* Evaluation of other distance metrics beside euclidean distance
* Evaluation of more NSA-based training algorithms in literature
* Evaluation of alternate testing algorithms
* Graphical results outside algorithms

Contributions
-------
* Feel free to bring contributions

Credits
-------

This code was originally developed in MATLAB, it was converted to Python with the aid of ChatGPT with some refactoring


