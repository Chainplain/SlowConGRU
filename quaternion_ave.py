import numpy as np

def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    """Normalize a single quaternion."""
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q)
    if norm == 0:
        raise ValueError("Zero-norm quaternion cannot be normalized.")
    return q / norm


def average_quaternions_eigen(quaternions: np.ndarray) -> np.ndarray:
    """
    Compute the cycle-averaged quaternion using the principal eigenvector method.

    Parameters
    ----------
    quaternions : ndarray of shape (N, 4)
        Array of N quaternions [w, x, y, z].
        Each quaternion does not need to be normalized.

    Returns
    -------
    q_avg : ndarray of shape (4,)
        The normalized average quaternion.
    """

    Q = np.asarray(quaternions, dtype=float)

    if Q.ndim != 2 or Q.shape[1] != 4:
        raise ValueError("Input must be of shape (N, 4)")

    # Normalize all quaternions first
    Q = Q / np.linalg.norm(Q, axis=1, keepdims=True)

    # Optional but recommended: enforce sign consistency
    # (avoid cancellation due to q and -q)
    for i in range(1, len(Q)):
        if np.dot(Q[0], Q[i]) < 0:
            Q[i] = -Q[i]

    # Build symmetric accumulator matrix M
    M = np.zeros((4, 4))
    for q in Q:
        M += np.outer(q, q)

    # Eigen decomposition (symmetric → use eigh)
    eigenvalues, eigenvectors = np.linalg.eigh(M)

    # Select eigenvector with largest eigenvalue
    v_max = eigenvectors[:, np.argmax(eigenvalues)]

    # Normalize result
    q_avg = normalize_quaternion(v_max)

    return q_avg


if __name__ == "__main__":
    # Generate sample quaternions around identity
    np.random.seed(0)

    N = 40
    noise = 0.1 * np.random.randn(N, 4)
    quats = np.tile([1, 0, 0, 0], (N, 1)) + noise
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(6, 8), sharex=True)
    component_labels = ["w", "x", "y", "z"]

    for idx, ax in enumerate(axes):
        ax.plot(quats[:, idx], marker="o", linestyle="-", alpha=0.7)
        ax.set_ylabel(component_labels[idx])
        ax.grid(True, linestyle=":", linewidth=0.5)

    axes[-1].set_xlabel("Sample index")
    plt.suptitle("Quaternion Components")
    plt.tight_layout()
    plt.show()
    
    q_slow = average_quaternions_eigen(quats)

    print("Averaged quaternion:", q_slow)
    print("Norm:", np.linalg.norm(q_slow))