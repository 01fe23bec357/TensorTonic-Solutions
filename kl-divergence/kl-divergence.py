import numpy as np

def kl_divergence(p, q, eps=1e-12):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    # Add epsilon to q for numerical stability
    q = q + eps

    # Compute KL divergence (ignore terms where p == 0)
    kl = np.sum(np.where(p > 0, p * np.log(p / q), 0.0))

    return kl