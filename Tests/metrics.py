import ot
import numpy as np

# X_true: shape (M, N) - True graph signals
# X_gen:  shape (M, N) - Flow Matching generated signals
def wasserstein_distance(X_true, X_gen):
    # 1. Compute pairwise squared Euclidean distance matrix
    M = ot.dist(X_gen, X_true, metric='sqeuclidean')

    # 2. Assume uniform weights for empirical samples
    a, b = np.ones(M.shape[0]) / M.shape[0], np.ones(M.shape[1]) / M.shape[1]

    # 3. Compute W2 squared (exact)
    w2_squared = ot.emd2(a, b, M)
    w2 = np.sqrt(w2_squared)
    return w2
#maybe compute Wasserstein Distance Over the Cycle Graph (with graph cost)
# would Gromov-Wasserstein be more appropriate for comparing distributions over graphs if we don't have equivariant archotecture?
