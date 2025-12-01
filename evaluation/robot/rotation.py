import numpy as np
from scipy.spatial.transform import Rotation as R


def rotation_6d_to_matrix(rot_6d: np.ndarray) -> np.ndarray:
    """Convert 6D rotation (two stacked 3D vectors) to a 3x3 matrix via Gram-Schmidt."""

    assert rot_6d.shape[-1] == 6, "Expected 6D rotation"
    a1 = rot_6d[..., 0:5:2]
    a2 = rot_6d[..., 1:6:2]

    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    dot = np.sum(b1 * a2, axis=-1, keepdims=True)
    proj = dot * b1
    b2 = a2 - proj
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)

    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def abs_6d_2_abs_euler(action: np.ndarray) -> np.ndarray:
    """Convert absolute 6D rotation action (left+right) to xyz+rpy+gripper."""

    left_xyz = action[0:3]
    left_6d = action[3:9]
    left_grip = action[9]

    right_xyz = action[10:13]
    right_6d = action[13:19]
    right_grip = action[19]

    left_matrix = rotation_6d_to_matrix(left_6d)
    right_matrix = rotation_6d_to_matrix(right_6d)

    left_euler = R.from_matrix(left_matrix).as_euler("xyz", degrees=False)
    right_euler = R.from_matrix(right_matrix).as_euler("xyz", degrees=False)

    return np.concatenate(
        [
            left_xyz,
            left_euler,
            [left_grip],
            right_xyz,
            right_euler,
            [right_grip],
        ]
    )


def rotation_matrix_to_6d(matrix: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to 6D representation (front two columns)."""

    return np.concatenate([matrix[0, :2], matrix[1, :2], matrix[2, :2]])


__all__ = ["rotation_6d_to_matrix", "rotation_matrix_to_6d", "abs_6d_2_abs_euler"]
