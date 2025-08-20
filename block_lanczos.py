import numpy as np

def block_lanczos(mat, max_iter=1000):
    """
    Solve mat * x = 0 over GF(2) using block Lanczos.
    mat: numpy array of shape (m,n) with dtype=uint8
    """
    m, n = mat.shape
    v = np.random.randint(0, 2, size=(n, 1), dtype=np.uint8)
    basis = []

    for it in range(max_iter):
        w = (mat @ v) % 2
        if not np.any(w):
            return v
        # orthogonalize vs previous
        for b in basis:
            if np.any((w ^ b) == 0):
                w = (w + b) % 2
        basis.append(w)
        v = np.random.randint(0, 2, size=(n, 1), dtype=np.uint8)

    return None
