import statistics as stats

import numpy as np

from .utils import EuclideanDistance as eucdist


def match_detector(trackers, samples, distance_metric=eucdist, verbose=0):
    """ Negative Selection matching function.
    This code implements the testing phase of the Negative Selection Algorithm
    which consist of matching trackers to those samples to be classified as
    nonself data, otherwise, if not match any of anomaly trackers, the sample
    is classified as a self (normal) data. It is developed to be applied after
    training phase with any variation of negative selection method, except
    when there are some specificities for test phase. Requires both testing
    samples and generated trackers from training phase.

    Parameters
    ----------
    distance_metric : 'euclidean' or custom, default='euclidean'
        Defines the distance metric to be adopted in order to measure distance
        between trackers and samples or between two detectors.
    verbose : int, default=0
        Enable verbose output. Note that this setting takes advantage of a
        per-process runtime setting in liblinear that, if enabled, may not work
        properly in a multithreaded context.
    Attributes
    ----------
    D : ndarray of shape (num_trackers, num_dimensions) \
        where num_trackers < max_trackers
        Corresponds to the set of unmatched trackers.
    coverage_index : 0 < float < f(coverage_factor)
        Detector coverage rate of current detectors allocated.
    self_matched : int,  0 < self_matched < num_attempts
        Counter of matched trackers during algorithm iterations.
    sample_distance : float
        Distance of a tracker to a self sample.
    coverage : boolean
        Defines if a tracker has matched another tracker
    num_attempts : int
        Maximum number of attempts to insert a tracker into the set.
    x : ndarray of shape (1, `num_dimensions`)
        A potential tracker which is exposed to the self samples.
    r : float
        Radius of the potential tracker, calculated based on self samples.
    See Also
    --------
    ClassicalNegativeSelection : Implementation of constant-radius NSA:
        The classical approach of the Negative Selection Algorithm with
        constant radius trackers.
    EuclideanDistance : A sample distance metric which is applied to self-shape
        evaluation of samples and trackers.
    Version
    --------
    v0.5 : Initial release
        Provided two approaches of NSA-based algorithms for testing phase and
        one matching function for testing phase. Euclidean distance is applied
        to compare trackers and samples in the shape space of features.
        Results are displayed in a matplotlib graph if verbose is set to 1 and
        for bidimensional datasets.
        This algorithm is applied only for real-valued problems for now.
    Notes
    -----
    This algorithm was developed in order to provide an implementation of
    negative selection-based methods for Python. This algorithm is part
    of the Artificial Immune Systems / Immune-inspired System family of
    methods and has a historical value in literature, as well as in
    the academic community. Feel free to apply this code at prominent
    anomaly detection datasets, but note that this approach is far
    from perfect and may not work as expected when applied to real world
    problems, as there are better methods which can achieve better results.
    If you intend to apply this in such problems anyway, do it at your risk!

    This code was initially developed for MATLAB during a PhD Thesis study,
    it was converted to Python with the aid of ChatGPT and then refactored
    into the resulting code.
    References
    ----------
    `Gonzalez, F., Dasgupta, D., & Niño, L. F. (2003, September). A randomized
    real-valued negative selection algorithm. In International Conference on
    Artificial Immune Systems (pp. 261-272). Berlin, Heidelberg: Springer
    Berlin Heidelberg.`__
    `Ji, Z., & Dasgupta, D. (2009). V-detector: An efficient negative selection
    algorithm with “probably adequate” detector coverage. Information Sciences,
    179(10), 1390–1406.`__
    Examples
    --------
    To be described...
    """

    # check if both sample and tracking data is present
    if samples is None:
        raise Exception("Testing samples must be available for NSA!")
    if trackers is None:
        raise Exception("Detection trackers must be available!")

    # Extract information from samples and trackers
    num_samples, num_dimensions = samples.shape
    num_trackers, num_elements = trackers.shape

    # check if tracking data have their radius
    if num_elements <= num_dimensions:
        raise Exception("Detection trackers have their radius missing!")

    # Set relevant information
    radius = num_elements - 1
    nonself = np.zeros(num_samples)
    minimum_distance = np.zeros(num_samples)
    location = np.zeros(num_samples)

    # Evaluate all testing samples
    for k in range(num_samples):
        detector_distance = np.zeros(num_trackers)

        # Check all trackers
        for d in range(num_trackers):
            detector_distance[d] = distance_metric(
                trackers[d, :num_dimensions], samples[k]
            )

        # Get the nearest tracker from sample
        min_location = np.argmin(detector_distance)
        minimum_distance[k] = detector_distance[min_location]
        location[k] = min_location

        # Time to classify samples
        predicted = 0
        if (
            minimum_distance[k] - trackers[min_location, radius] <= 0
            or minimum_distance[k] + trackers[min_location, radius] <= 0
        ):
            predicted = 1

        # Set labels (0 - normal and 1- anomalous)
        nonself[k] = predicted > 0

    if verbose == 1:
        print("=== Testing ===")
        print("Number of self samples detected: ", np.sum(nonself == 0))
        print("Number of nonself samples detected: ", np.sum(nonself > 0))
        print("Average distance: ", np.mean(minimum_distance))
        print("Most matched tracker: ", int(stats.mode(location) + 1))

    return nonself, minimum_distance, location
