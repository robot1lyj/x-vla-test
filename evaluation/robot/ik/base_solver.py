"""
Pinocchio + CasADi based IK solver (no ROS/visualization), simplified.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple, Dict, Any
import os
import numpy as np
import casadi
import pinocchio as pin
from pinocchio import casadi as cpin
from pinocchio.robot_wrapper import RobotWrapper


def rpy_to_quat(roll: float, pitch: float, yaw: float) -> pin.Quaternion:
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    w = cr * cp * cy - sr * sp * sy
    x = sr * cp * cy + cr * sp * sy
    y = cr * sp * cy - sr * cp * sy
    z = cr * cp * sy + sr * sp * cy
    return pin.Quaternion(w, x, y, z)


def xyzrpy_to_se3(x: float, y: float, z: float, roll: float, pitch: float, yaw: float) -> pin.SE3:
    quat = rpy_to_quat(roll, pitch, yaw)
    return pin.SE3(quat, np.array([x, y, z], dtype=float))


class BaseArmIK:
    def __init__(
        self,
        urdf_path: str,
        joints_to_lock: Optional[Iterable[str]] = None,
        add_ee_on_joint: str = "joint6",
        add_ee_translation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        add_ee_rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        weight_pose: float = 20.0,
        weight_reg: float = 0.01,
        smooth_weight: float = 0.0,
        max_iter: int = 50,
        tol: float = 1e-4,
        trust_region: Optional[float] = None,
        solver_opts: Optional[Dict[str, Any]] = None,
        enable_collision: bool = False,
    ) -> None:
        if not os.path.isfile(urdf_path):
            raise FileNotFoundError(f"URDF not found: {urdf_path}")

        urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
        package_dirs = [urdf_dir]

        if joints_to_lock is None:
            joints_to_lock = ["joint7", "joint8"]

        self.robot: RobotWrapper = RobotWrapper.BuildFromURDF(urdf_path, package_dirs=package_dirs)
        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=list(joints_to_lock),
            reference_configuration=np.zeros(self.robot.model.nq),
        )
        self.nq = self.reduced_robot.model.nq
        self._joint_lower = self.reduced_robot.model.lowerPositionLimit.copy()
        self._joint_upper = self.reduced_robot.model.upperPositionLimit.copy()

        # Add EE frame for convenience.
        ee_quat = rpy_to_quat(*add_ee_rpy)
        self.reduced_robot.model.addFrame(
            pin.Frame(
                "ee",
                self.reduced_robot.model.getJointId(add_ee_on_joint),
                pin.SE3(ee_quat, np.array(add_ee_translation, dtype=float)),
                pin.FrameType.OP_FRAME,
            )
        )
        self.robot.data = self.robot.model.createData()
        self.reduced_robot.data = self.reduced_robot.model.createData()

        # Collision (optional).
        self._collision_enabled = False
        self.geom_model = None
        self.geometry_data = None
        if enable_collision:
            try:
                self.geom_model = pin.buildGeomFromUrdf(
                    self.robot.model,
                    urdf_path,
                    pin.GeometryType.COLLISION,
                    package_dirs=package_dirs,
                )
                self.geometry_data = pin.GeometryData(self.geom_model)
                self._collision_enabled = True
            except Exception:
                self.geom_model = None
                self.geometry_data = None
                self._collision_enabled = False

        # CasADi model.
        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()
        self.cq = casadi.SX.sym("q", self.nq, 1)
        self.cTf = casadi.SX.sym("tf", 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)
        self.ee_fid = self.reduced_robot.model.getFrameId("ee")

        log_vec = cpin.log6(self.cdata.oMf[self.ee_fid].inverse() * cpin.SE3(self.cTf)).vector
        weights = casadi.diag(casadi.vertcat(*([weight_pose] * 6)))
        weighted = weights @ log_vec
        self.error_fun = casadi.Function("error", [self.cq, self.cTf], [weighted])

        # Opti
        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.nq)
        self.param_tf = self.opti.parameter(4, 4)
        self.param_q_last = self.opti.parameter(self.nq)

        tracking_cost = casadi.sumsqr(self.error_fun(self.var_q, self.param_tf))
        reg_cost = weight_reg * casadi.sumsqr(self.var_q)
        smooth_cost = smooth_weight * casadi.sumsqr(self.var_q - self.param_q_last)
        self.opti.subject_to(
            self.opti.bounded(self.reduced_robot.model.lowerPositionLimit, self.var_q, self.reduced_robot.model.upperPositionLimit)
        )
        self.param_step_lower = None
        self.param_step_upper = None
        if trust_region is not None:
            self.param_step_lower = self.opti.parameter(self.nq)
            self.param_step_upper = self.opti.parameter(self.nq)
            self.opti.subject_to(self.param_step_lower <= self.var_q)
            self.opti.subject_to(self.var_q <= self.param_step_upper)

        self.opti.minimize(tracking_cost + reg_cost + smooth_cost)
        opts = {"ipopt": {"print_level": 0, "max_iter": max_iter, "tol": tol}, "print_time": False}
        if solver_opts:
            opts["ipopt"].update(solver_opts.get("ipopt", {}))
        self.opti.solver("ipopt", opts)

        self.q_seed = np.zeros(self.nq)
        self.q_last = np.zeros(self.nq)

    def solve(self, target: pin.SE3 | np.ndarray, trust_region: Optional[float] = None, check_collision: bool = True) -> Tuple[Optional[np.ndarray], bool, str]:
        if isinstance(target, pin.SE3):
            T = target.homogeneous
        else:
            T = np.asarray(target, dtype=float)
            if T.shape != (4, 4):
                raise ValueError("target must be SE3 or (4,4) array")

        self.opti.set_initial(self.var_q, self.q_seed)
        self.opti.set_value(self.param_tf, T)
        self.opti.set_value(self.param_q_last, self.q_last)

        if self.param_step_lower is not None and self.param_step_upper is not None:
            tr = trust_region if trust_region is not None else np.inf
            lower = np.maximum(self._joint_lower, self.q_last - tr)
            upper = np.minimum(self._joint_upper, self.q_last + tr)
            self.opti.set_value(self.param_step_lower, lower)
            self.opti.set_value(self.param_step_upper, upper)

        try:
            self.opti.solve_limited()
            q = np.asarray(self.opti.value(self.var_q)).reshape(-1)
            self.q_last = q.copy()
            self.q_seed = q.copy()
            success = True
            info = "ok"
            if check_collision and self.is_self_collision(q):
                success = False
                info = "self-collision detected"
            return q, success, info
        except Exception as exc:  # pylint: disable=broad-except
            return None, False, f"solve failed: {exc}"

    def is_self_collision(self, q: np.ndarray, gripper: Optional[np.ndarray] = None) -> bool:
        if not self._collision_enabled or self.geom_model is None or self.geometry_data is None:
            return False
        if gripper is None:
            gripper = np.zeros(2)
        q_full = np.concatenate([q.reshape(-1), gripper.reshape(-1)], axis=0)
        pin.forwardKinematics(self.robot.model, self.robot.data, q_full)
        pin.updateGeometryPlacements(self.robot.model, self.robot.data, self.geom_model, self.geometry_data)
        return bool(pin.computeCollisions(self.geom_model, self.geometry_data, False))

    def forward_k(self, q: np.ndarray) -> np.ndarray:
        pin.forwardKinematics(self.reduced_robot.model, self.reduced_robot.data, q)
        pose = self.reduced_robot.data.oMf[self.reduced_robot.model.getFrameId("ee")]
        pos = pose.translation
        euler = pin.rpy.matrixToRpy(pose.rotation)
        return np.concatenate([pos, euler], axis=0)


__all__ = ["BaseArmIK", "rpy_to_quat", "xyzrpy_to_se3"]
