# example of multi-class classification task
from numpy import where
from sklearn.datasets import make_blobs

from evalNSA import evalNSA

# define dataset
X, y = make_blobs(n_samples=1000, centers=4, random_state=1)

# Training based on 'normal' data
self_data = where(y == 0)[0]
train = self_data[self_data < 500]

# Test all other data
test = [i for i in range(500, 1000)]
# Generate samples
Xtr = X[train]
Xts = X[test]

# Algorithmic parameters
radius = 0.2
threshold = 0.4
probes = 100
trials = 100
approach = 2

# Perform execution
ypred = evalNSA(Xtr, Xts, approach, radius, threshold, probes, trials, verbose=1)
