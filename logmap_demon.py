# quaternion_logmap_demo.py
# Demonstrates the quaternion logarithmic map r = log(q) (minimal 3D tangent/rotation-vector)
# and shows it is invertible via the exponential map q = exp(r) for unit quaternions.

import numpy as np


def normalize_quat(q: np.ndarray) -> np.ndarray:
    """Normalize quaternion to unit length. Quaternion format: [w, x, y, z]."""
    q = np.asarray(q, dtype=float).reshape(4)
    n = np.linalg.norm(q)
    if n == 0:
        raise ValueError("Zero quaternion cannot be normalized.")
    return q / n


def quat_conj(q: np.ndarray) -> np.ndarray:
    """Quaternion conjugate."""
    q = np.asarray(q, dtype=float).reshape(4)
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product. Both in [w, x, y, z]."""
    w1, x1, y1, z1 = np.asarray(q1, dtype=float).reshape(4)
    w2, x2, y2, z2 = np.asarray(q2, dtype=float).reshape(4)
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def quat_to_rotvec_log(q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Logarithmic map for a UNIT quaternion q -> rotation vector r in R^3.
    If q = [cos(theta/2), u*sin(theta/2)], then log(q) = u*theta (a 3D rotation vector).

    Returns r with principal angle theta in (-pi, pi].
    """
    q = normalize_quat(q)
    w = np.clip(q[0], -1.0, 1.0)
    v = q[1:]
    vnorm = np.linalg.norm(v)

    if vnorm < eps:
        # Near identity: theta ~ 0, so r ~ 0
        return np.zeros(3)

    # Robust rotation angle: theta = 2*atan2(||v||, w)
    theta = 2.0 * np.arctan2(vnorm, w)

    # Wrap to principal angle (-pi, pi]
    if theta > np.pi:
        theta -= 2.0 * np.pi

    # r = u * theta, with u = v / sin(theta/2) = v / ||v|| (for unit q, ||v|| = |sin(theta/2)|)
    r = (theta / vnorm) * v
    return r


def rotvec_to_quat_exp(r: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Exponential map for rotation vector r in R^3 -> UNIT quaternion q.
    If r = u*theta, then exp(r) = [cos(theta/2), u*sin(theta/2)].
    """
    r = np.asarray(r, dtype=float).reshape(3)
    theta = np.linalg.norm(r)

    if theta < eps:
        return np.array([1.0, 0.0, 0.0, 0.0])

    u = r / theta
    half = 0.5 * theta
    q = np.array([np.cos(half), *(u * np.sin(half))])
    return normalize_quat(q)


def quat_distance(q1: np.ndarray, q2: np.ndarray) -> float:
    """
    Distance between rotations represented by unit quaternions, accounting for q ~ -q.
    We use the absolute dot product: angle_err = 2*acos(|<q1,q2>|).
    """
    q1 = normalize_quat(q1)
    q2 = normalize_quat(q2)
    d = abs(float(np.dot(q1, q2)))
    d = np.clip(d, -1.0, 1.0)
    return 2.0 * np.arccos(d)


def random_unit_vector(rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=3)
    n = np.linalg.norm(v)
    return v / n


def demo_once(theta: float, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    axis = random_unit_vector(rng)
    r_true = axis * theta

    q = rotvec_to_quat_exp(r_true)
    r = quat_to_rotvec_log(q)
    q_recon = rotvec_to_quat_exp(r)

    print("=== Demo ===")
    print(f"theta (rad): {theta:.6f}")
    print(f"axis: {axis}")
    print(f"r_true: {r_true}")
    print(f"q:      {q}")
    print(f"log(q): {r}")
    print(f"exp(log(q)): {q_recon}")
    print(f"rotation error (rad): {quat_distance(q, q_recon):.3e}")
    print(f"r error norm: {np.linalg.norm(r - r_true):.3e}\n")


def main():
    # Show invertibility on a few angles (avoid exactly pi to dodge axis ambiguity there)
    demo_once(theta=0.0, seed=1)
    demo_once(theta=0.3, seed=2)
    demo_once(theta=1.2, seed=3)
    demo_once(theta=3.0, seed=4)  # < pi, still fine

    # Random stress test
    rng = np.random.default_rng(123)
    max_rot_err = 0.0
    for _ in range(200):
        axis = random_unit_vector(rng)
        # sample angle in (-pi, pi)
        theta = (rng.random() * 2.0 - 1.0) * (np.pi - 1e-3)
        r = axis * theta
        q = rotvec_to_quat_exp(r)
        r2 = quat_to_rotvec_log(q)
        q2 = rotvec_to_quat_exp(r2)
        err = quat_distance(q, q2)
        max_rot_err = max(max_rot_err, err)

    print(f"Max rotation error over 200 random tests (rad): {max_rot_err:.3e}")
    print("If this is ~1e-12 to 1e-9, exp(log(q)) is numerically invertible as expected.")


if __name__ == "__main__":
    main()