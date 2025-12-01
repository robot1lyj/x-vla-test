from typing import Optional, Tuple

import numpy as np
import pinocchio as pin

from evaluation.robot.ik.base_solver import BaseArmIK, xyzrpy_to_se3


class PiperIKSolver(BaseArmIK):
    def __init__(
        self,
        urdf_path: str,
        weight_pose: float = 20.0,
        weight_reg: float = 0.01,
        smooth_weight: float = 0.0,
        max_iter: int = 50,
        tol: float = 1e-4,
        jump_threshold_rad: float = 30.0 / 180.0 * np.pi,
        trust_region: Optional[float] = None,
    ):
        super().__init__(
            urdf_path=urdf_path,
            joints_to_lock=["joint7", "joint8"],
            add_ee_on_joint="joint6",
            add_ee_translation=(0.0, 0.0, 0.0),
            add_ee_rpy=(0.0, 0.0, 0.0),
            weight_pose=weight_pose,
            weight_reg=weight_reg,
            smooth_weight=smooth_weight,
            max_iter=max_iter,
            tol=tol,
            trust_region=trust_region,
            enable_collision=False,
        )
        self.jump_threshold_rad = jump_threshold_rad
        self.history_data: Optional[np.ndarray] = None

    def solve_pose(self, target_xyzrpy: np.ndarray, q_init: Optional[np.ndarray] = None) -> Tuple[np.ndarray, bool, bool]:
        if target_xyzrpy.shape[-1] != 6:
            raise ValueError("Expected [x,y,z,roll,pitch,yaw]")
        target = xyzrpy_to_se3(*target_xyzrpy.tolist())
        if q_init is not None:
            self.q_seed = q_init.copy()
        q, success, info = super().solve(target, trust_region=None, check_collision=True)
        if q is None:
            return np.zeros(self.nq), False, True

        collision = "collision" in info.lower()
        if self.history_data is not None:
            max_diff = np.max(np.abs(self.history_data - q))
            if max_diff > self.jump_threshold_rad:
                self.q_seed = np.zeros_like(q)
            else:
                self.q_seed = q.copy()
        else:
            self.q_seed = q.copy()
        self.history_data = q.copy()
        return q, success, collision

    def forward_k(self, q: np.ndarray) -> np.ndarray:
        return super().forward_k(q)


__all__ = ["PiperIKSolver"]
