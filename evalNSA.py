import matplotlib.pyplot as plt
import numpy as np

from NegativeSelection import ClassicalNegativeSelection, VDetector, match_detector


def evalNSA(
    training,
    validation,
    type=1,
    self_radius=0.1,
    thresold=0.4,
    num_detectors=300,
    num_trials=200,
    verbose=0,
):
    """ Negative Selection algorithm evaluator.
    This code implements the Negative Selection Algorithm in both training and
    test phases which respectively generate trackers based on training normal
    data and apply them to test samples which contains both normal and
    anomalous samples.

    Parameters
    ----------
    type : int
        Defines the training algorithm to be applied, note that for the
        classical NSA execution, the constant radius should be included
        into the trackers data, as the test phase requires this information.
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
        Counter of matched trackers during iterations of the algorithm.
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
    VDetector : Implementation of Variable-radius detector generation:
        Another approach of the Negative Selection Algorithm which is improved
        with a tracker overlapping and coverage checking during the allocation
        of trackers in order to avoid redundancy.
    match_detector : Implementation of testing phase of Negative Selection:
        After generating the trackers, test samples are evaluated and
        classified as normal or anomalous samples based on their distance
        to a nearest detector in shape space.
    EuclideanDistance : A sample distance metric which is applied to
        self-shapeevaluation of samples and trackers.
    Version
    --------
    v0.5 : Initial release
        Provided two approaches of NSA-based algorithms for testing phase and
        one matching function for testing phase. Euclidean distance is applied
        to compare trackers and samples in the shape space of features. Results
        are displayed in a matplotlib graph if verbose is set to 1 and for
        bidimensional datasets.
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
    `Gonzalez, F., Dasgupta, D., & Niño, L. F. (2003, September). A
    randomized real-valued negative selection algorithm. In International
    Conference on Artificial Immune Systems (pp. 261-272). Berlin,
    Heidelberg: Springer Berlin Heidelberg.`__
    `Ji, Z., & Dasgupta, D. (2009). V-detector: An efficient negative
    selection algorithm with “probably adequate” detector coverage.
    Information Sciences, 179(10), 1390–1406.`__
    Examples
    --------
    To be described...
    """

    if training is None or validation is None:
        raise Exception("NSA must have training and testing samples available!")

    # Training procedure
    if type == 1:  # Classical NSA
        trackers, matched = ClassicalNegativeSelection(
            training, self_radius, thresold, num_detectors, num_trials, verbose=verbose
        )
        trackers = np.array([np.append(x, thresold) for x in trackers])
    elif type == 2:  # V-Detector
        trackers, matched = VDetector(
            training, self_radius, num_detectors, 0.975, num_trials, verbose=verbose
        )
    else:  # No method found
        raise Exception("No NSA method found, please set a valid type!")

    # Preparing for test
    num_samples, num_dimensions = validation.shape

    # Testing procedure and results
    unmatched, min_distance, detector_id = match_detector(
        trackers, validation, verbose=verbose
    )

    # Plotting results as a fault detection problem
    if (num_dimensions == 2) and (verbose == 1):
        fig, axs = plt.subplots(2, 1, figsize=(8, 6))
        axs[0].plot(min_distance, "r-o")
        axs[0].set_title("Distance with a Detector")
        axs[0].set_xlabel("time (ms)")
        axs[0].set_ylabel("distance")

        axs[1].plot(unmatched, "k-*")
        axs[1].set_title("System state (0-Normal/1-Fault)")
        axs[1].set_xlabel("time (ms)")
        axs[1].set_ylabel("Status")

        plt.tight_layout()
        plt.show()

    return unmatched
