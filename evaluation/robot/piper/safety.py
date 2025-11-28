import numpy as np
from typing import Tuple


def clamp_to_limits(q: np.ndarray, lower: np.ndarray, upper: np.ndarray, margin: float = 0.0) -> np.ndarray:
    """Clamp joint angles into limits with optional margin."""
    l = lower + margin
    u = upper - margin
    return np.clip(q, l, u)


def limit_velocity(q_prev: np.ndarray, q_next: np.ndarray, max_step: float) -> Tuple[np.ndarray, bool]:
    """Limit per-joint step to avoid large jumps."""
    diff = q_next - q_prev
    clipped = np.clip(diff, -max_step, max_step)
    limited = q_prev + clipped
    return limited, bool(np.any(np.abs(diff) > max_step))

