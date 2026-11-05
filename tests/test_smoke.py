"""Import + math-only sanity checks. Run with:  python -m pytest tests/  or  python tests/test_smoke.py"""
import numpy as np

from src.utils.transforms import make_T, invert_T, compose, pose_error


def test_invert_T_roundtrips():
    rng = np.random.default_rng(0)
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    theta = 0.7
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K
    t = rng.normal(size=3)
    T = make_T(R, t)
    assert np.allclose(compose(T, invert_T(T)), np.eye(4), atol=1e-9)


def test_pose_error_zero_for_identical():
    T = make_T(np.eye(3), np.array([1.0, 2.0, 3.0]))
    t_err, r_err = pose_error(T, T)
    assert t_err < 1e-9 and r_err < 1e-9


def test_pose_error_translation():
    T1 = make_T(np.eye(3), np.array([0.0, 0.0, 0.0]))
    T2 = make_T(np.eye(3), np.array([0.3, 0.4, 0.0]))
    t_err, _ = pose_error(T1, T2)
    assert abs(t_err - 0.5) < 1e-9


if __name__ == "__main__":
    test_invert_T_roundtrips()
    test_pose_error_zero_for_identical()
    test_pose_error_translation()
    print("OK")
