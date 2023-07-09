import matplotlib.pyplot as plt
import numpy as np

from .utils import EuclideanDistance as eucdist


def VDetector(
    samples,
    self_radius,
    max_trackers,
    coverage_factor,
    max_attempts,
    evaluate_distance=eucdist,
    verbose=0,
):
    """ `V-detector` - Variable-radius Negative Selection detector generation.
    This algorithm improves the Real-valued Negative Selection Algorithm by
    providing trackers with variable radius size, as well as a coverage rate
    checking, also being able to check if trackers are overlapped. As the
    classical NSA, this method is developed for real-valued anomaly detection
    problems based on the shape space of features, the detectors which does
    not match with sample data and with less overlaps will compose the set
    of trackers. Also applied to training phase.

    Parameters
    ----------
    self_radius : float, default=1e-2
        Specifies the radius of a self sample based on shape-space
        representation. This radius defines the self (normal data) margin,
        in which a tracker, when presented to these samples, is marked to
        be eliminated if it matches these samples within this space.
    max_trackers : int, default=100
        Specifies the number of trackers to be allocated within the
        shape-space.
    coverage_factor : float, default=0.97
        Detector coverage rate, determines if how detector radius can be
        covered by another detector.
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
    match_detector : Implementation of testing phase of Negative Selection:
        After generating the trackers, test samples are evaluated and
        classified as normal or anomalous samples based on their distance
        to a nearest detector in shape space.
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
    `Ji, Z., & Dasgupta, D. (2009). V-detector: An efficient negative selection
    algorithm with “probably adequate” detector coverage. Information Sciences,
    179(10), 1390–1406.`__
    Examples
    --------
    To be described...
    """

    # check if sample data is present
    if samples is None:
        raise Exception("Training samples must be available for NSA!")

    num_samples, num_dimensions = samples.shape
    # Empty array to store detected points (x, r)
    D = np.empty((0, num_dimensions + 1))
    # Counter for non-detected points
    self_matched = 0
    coverage_index = 0
    num_attempts = 0

    while (
        (len(D) < max_trackers)
        and (coverage_index < (1 / (1 - coverage_factor)))
        and (num_attempts < max_attempts)
    ):
        # Initialize values
        coverage = False
        r = np.inf
        # define shapespace min and max before starting
        x = 15 * np.random.rand(num_dimensions) - np.random.rand(num_dimensions)

        # Verify if the new detector overlaps another one, if any
        for k in range(len(D)):
            detector_distance = evaluate_distance(x, D[k, :num_dimensions])
            if detector_distance <= D[k, num_dimensions]:
                coverage_index += 1
                coverage = True
                break  # No need to continue checking other points

        # Verify distance of the non-overlapping detector and sampling data
        if not coverage:
            sample_distance = np.zeros(num_samples)
            for k in range(num_samples):
                sample_distance[k] = evaluate_distance(x, samples[k])

            # Tangent radius calculation based on the nearest training sample
            s = np.min(sample_distance)
            if s - self_radius <= r:
                r = s - self_radius

            if r > self_radius:
                # the detector can be assigned as a true detector (nonself)
                D = np.vstack([D, np.concatenate([x, [r]])])
                num_attempts = 0
            else:
                # detector has matched a training sample
                self_matched += 1
                num_attempts += 1
        else:
            num_attempts += 1

    # Plotting results for 2-dimensions problem (for testing purposes)
    if num_dimensions == 2 and verbose == 1:
        t = np.linspace(0, 2 * np.pi, 1000)
        if D.shape[1] == 3:
            plt.figure()
            for ss in range(num_samples):
                xs = self_radius * np.cos(t) + samples[ss, 0]
                ys = self_radius * np.sin(t) + samples[ss, 1]
                plt.plot(xs, ys, "r-")
                plt.plot(samples[ss, 0], samples[ss, 1], "rd")

            for k in range(len(D)):
                plt.plot(D[k, 0], D[k, 1], "bx")
                x = D[k, 2] * np.cos(t) + D[k, 0]
                y = D[k, 2] * np.sin(t) + D[k, 1]
                plt.plot(x, y, "b-")

            plt.show()

    if verbose == 1:
        print("=== Training ===")
        print("Detectors generated: ", len(D))
        print("Detectors discarded: ", self_matched)
        print("Number of allocation attempts: ", num_attempts)
        print("Coverage rate: ", coverage_index)

    return D, self_matched
