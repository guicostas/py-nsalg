import matplotlib.pyplot as plt
import numpy as np

from .utils import EuclideanDistance as eucdist


def ClassicalNegativeSelection(
    samples,
    self_radius,
    detection_radius,
    max_trackers,
    max_attempts,
    evaluate_distance=eucdist,
    verbose=0,
):
    """ Classical Negative Selection detector generation.
    This is the Real-valued Negative Selection Algorithm, developed initially
    for anomaly detection problems, in which we have a shape space of features
    and we set trackers (detectors) in order to match anomaly data based on
    normal behavior, whose data cannot be matched, otherwise the detector will
    be discarded. The detectors which does not match with sample data (i.e.
    self samples), will compose the set of trackers. This step corresponds to
    the training phase of the algorithm.

    Parameters
    ----------
    self_radius : float, default=1e-2
        Specifies the radius of a self sample based on shape-space
        representation. This radius defines the self (normal data) margin, in
        which a tracker, when presented to these samples, is marked to be
        eliminated if it matches these samples within this space.
    detection_radius : float, default=4e-2
        Specifies the radius of a nonself sample for constant radius-based
        methods. When a tracker is generated in shape-space, this value is
        assigned as a radius in order to detect a test sample based on its
        distance to the tracker.
    max_trackers : int, default=100
        Specifies the number of trackers to be allocated within the
        shape-space.
    max_attempts : int, default=100
        Maximum attempts to allocate a tracker without matching a self sample,
        after this value is reached, i.e. if trackers match self sample, the
        algorithm stops and returns all trackers allocated, regardless of
        its maximum.
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
    self_matched : int,  0 < self_matched < num_attempts
        Counter of matched trackers during iterations of the algorithm.
    sample_distance : float
        Distance of a tracker to a self sample.
    matched : boolean
        Defines if a tracker has matched a self sample
    num_attempts : int
        Maximum number of attempts to insert a tracker into the set.
    x : ndarray of shape (1, `num_dimensions`)
        A potential tracker which is exposed to the self samples.
    See Also
    --------
    match_detector : Implementation of testing phase of Negative Selection:
        After generating the trackers, test samples are evaluated and
        classified as normal or anomalous samples based on their distance to
        a nearest detector in shape space.
    VDetector : Implementation of Variable-radius detector generation:
        Another approach of the Negative Selection Algorithm which is improved
        with a tracker overlapping and coverage checking during the allocation
        of trackers in order to avoid redundancy.
    EuclideanDistance : A sample distance metric which is applied to self-shape
        evaluation of samples and trackers.
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
    `Gonzalez, F., Dasgupta, D., & Niño, L. F. (2003, September). A randomized
    real-valued negative selection algorithm. In International Conference on
    Artificial Immune Systems (pp. 261-272). Berlin, Heidelberg: Springer
    Berlin Heidelberg.`__
    `Forrest, S., Perelson, A. S., Allen, L., & Cherukuri, R. (1994, May).
    Self-nonself discrimination in a computer. In Proceedings of 1994 IEEE
    computer society symposium on research in security and privacy
    (pp. 202-212). Ieee.`__
    Examples
    --------
    To be described...
    """

    # check if sample data is present
    if samples is None:
        raise Exception("Training samples must be available for NSA!")

    D = []  # List to store detected points
    self_matched = 0  # Counter for non-detected points
    num_samples, num_dimensions = samples.shape
    num_attempts = 0

    while len(D) < max_trackers and num_attempts < max_attempts:
        # define shapespace min and max before starting
        x = 5 * np.random.rand(num_dimensions) - np.random.rand(num_dimensions)
        matched = False

        # evaluate training data
        for k in range(num_samples):
            sample_distance = evaluate_distance(x, samples[k])

            if sample_distance < self_radius + detection_radius:
                matched = True
                break

        if matched:
            self_matched += 1
            num_attempts += 1
        else:
            D.append(x)
            num_attempts = 0

    D = np.array(D)

    # Plotting results for 2-dimensions problem (for testing purposes)
    if num_dimensions == 2 and verbose == 1:
        t = np.linspace(0, 2 * np.pi, 1000)
        if D.shape[1] == 2:
            plt.figure()
            for ss in range(num_samples):
                xs = self_radius * np.cos(t) + samples[ss, 0]
                ys = self_radius * np.sin(t) + samples[ss, 1]
                plt.plot(xs, ys, "r-")
                plt.plot(samples[ss, 0], samples[ss, 1], "rd")

            for k in range(len(D)):
                plt.plot(D[k, 0], D[k, 1], "bx")
                x = detection_radius * np.cos(t) + D[k, 0]
                y = detection_radius * np.sin(t) + D[k, 1]
                plt.plot(x, y, "b-")

            plt.show()

    if verbose == 1:
        print("=== Training ===")
        print("Detectors generated: ", len(D))
        print("Detectors discarded: ", self_matched)
        print("Number of allocation attempts: ", num_attempts)

    return D, self_matched
