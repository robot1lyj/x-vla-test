import logging
from typing import Optional, Tuple

import casadi
import numpy as np
import pinocchio as pin
from pinocchio import casadi as cpin
from pinocchio.robot_wrapper import RobotWrapper
from scipy.spatial.transform import Rotation as R


def xyzrpy_to_se3(xyzrpy: np.ndarray) -> np.ndarray:
    """Convert [x, y, z, roll, pitch, yaw] to 4x4 SE3 matrix."""
    assert xyzrpy.shape[-1] == 6
    rot = pin.rpy.rpyToMatrix(xyzrpy[3], xyzrpy[4], xyzrpy[5])
    return pin.SE3(rot, np.array(xyzrpy[:3])).homogeneous


class ArmIK:
    """
    Casadi-based IK wrapper, ported from the ROS node but without ROS dependencies.
    """

    def __init__(
        self,
        urdf_path: str,
        weight_pose: float = 20.0,
        weight_reg: float = 0.01,
        smooth_weight: float = 0.0,
        max_iter: int = 50,
        tol: float = 1e-4,
        jump_threshold_rad: float = 30.0 / 180.0 * 3.1415926,
        trust_region: Optional[float] = None,
    ):
        self.urdf_path = urdf_path
        self.jump_threshold_rad = jump_threshold_rad
        self.init_data: Optional[np.ndarray] = None
        self.history_data: Optional[np.ndarray] = None

        self.robot = RobotWrapper.BuildFromURDF(urdf_path)
        # Lock gripper joints if present.
        lock_joints = [j for j in ["joint7", "joint8"] if j in self.robot.model.names]
        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=lock_joints,
            reference_configuration=np.zeros(self.robot.model.nq),
        )
        # Add ee frame on joint6.
        q = pin.Quaternion(1, 0, 0, 0)
        self.reduced_robot.model.addFrame(
            pin.Frame(
                "ee",
                self.reduced_robot.model.getJointId("joint6"),
                pin.SE3(q, np.array([0.0, 0.0, 0.0])),
                pin.FrameType.OP_FRAME,
            )
        )

        # Collision model (kept for basic self-collision check).
        self.geom_model = pin.buildGeomFromUrdf(self.robot.model, urdf_path, pin.GeometryType.COLLISION)
        self.geometry_data = pin.GeometryData(self.geom_model)

        # Casadi setup.
        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()
        self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1)
        self.cTf = casadi.SX.sym("tf", 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)
        self.gripper_id = self.reduced_robot.model.getFrameId("ee")
        self.error = casadi.Function(
            "error",
            [self.cq, self.cTf],
            [
                casadi.vertcat(
                    cpin.log6(self.cdata.oMf[self.gripper_id].inverse() * cpin.SE3(self.cTf)).vector,
                )
            ],
        )
        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.reduced_robot.model.nq)
        self.var_q_last = self.opti.parameter(self.reduced_robot.model.nq)
        self.param_tf = self.opti.parameter(4, 4)
        totalcost = casadi.sumsqr(self.error(self.var_q, self.param_tf))
        regularization = casadi.sumsqr(self.var_q)
        smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last)
        self.opti.subject_to(
            self.opti.bounded(
                self.reduced_robot.model.lowerPositionLimit,
                self.var_q,
                self.reduced_robot.model.upperPositionLimit,
            )
        )
        self.opti.minimize(weight_pose * totalcost + weight_reg * regularization + smooth_weight * smooth_cost)
        # Optional trust region around seed (set per solve).
        self.trust_region = trust_region
        if trust_region is not None:
            self.param_step_lower = self.opti.parameter(self.reduced_robot.model.nq)
            self.param_step_upper = self.opti.parameter(self.reduced_robot.model.nq)
            self.opti.subject_to(self.param_step_lower <= self.var_q)
            self.opti.subject_to(self.var_q <= self.param_step_upper)
        else:
            self.param_step_lower = None
            self.param_step_upper = None
        opts = {
            "ipopt": {"print_level": 0, "max_iter": max_iter, "tol": tol},
            "print_time": False,
        }
        self.opti.solver("ipopt", opts)

    def solve(self, target_pose: np.ndarray, q_init: Optional[np.ndarray] = None) -> Tuple[np.ndarray, bool, bool]:
        """Return (q, success, collision)."""
        if q_init is None:
            q_init = self.init_data if self.init_data is not None else np.zeros(self.reduced_robot.model.nq)

        def _try(q_guess: np.ndarray):
            q_guess = np.clip(q_guess, self.reduced_robot.model.lowerPositionLimit, self.reduced_robot.model.upperPositionLimit)
            self.opti.set_initial(self.var_q, q_guess)
            self.opti.set_value(self.param_tf, target_pose)
            self.opti.set_value(self.var_q_last, q_guess)
            if self.trust_region is not None and self.param_step_lower is not None and self.param_step_upper is not None:
                lower = np.maximum(self.reduced_robot.model.lowerPositionLimit, q_guess - self.trust_region)
                upper = np.minimum(self.reduced_robot.model.upperPositionLimit, q_guess + self.trust_region)
                self.opti.set_value(self.param_step_lower, lower)
                self.opti.set_value(self.param_step_upper, upper)
            sol = self.opti.solve_limited()
            return self.opti.value(self.var_q)

        try:
            sol_q = _try(q_init)
        except Exception as exc:
            logging.error("IK solve failed: %s. Retrying from zeros.", exc)
            try:
                sol_q = _try(np.zeros(self.reduced_robot.model.nq))
            except Exception as exc2:
                logging.error("IK retry from zeros also failed: %s. Trying neutral seed.", exc2)
                try:
                    neutral = pin.neutral(self.reduced_robot.model)
                    sol_q = _try(neutral)
                except Exception as exc3:
                    logging.error("IK retry from neutral also failed: %s", exc3)
                    return q_init, False, True

        # Jump guard.
        if self.history_data is not None:
            max_diff = np.max(np.abs(self.history_data - sol_q))
            if max_diff > self.jump_threshold_rad:
                logging.warning("IK jump detected (%.2f deg), resetting init.", max_diff * 180.0 / np.pi)
                self.init_data = np.zeros_like(sol_q)
            else:
                self.init_data = sol_q
        else:
            self.init_data = sol_q
        self.history_data = sol_q

        collision = self.check_self_collision(sol_q)
        return sol_q, True, collision

    def check_self_collision(self, q: np.ndarray) -> bool:
        # Append zeros for locked joints so full model has consistent dimension.
        full_q = np.concatenate([q, np.zeros(self.robot.model.nq - q.shape[0])], axis=0)
        pin.forwardKinematics(self.robot.model, self.robot.data, full_q)
        pin.updateGeometryPlacements(self.robot.model, self.robot.data, self.geom_model, self.geometry_data)
        return pin.computeCollisions(self.geom_model, self.geometry_data, False)

    def forward_k(self, q: np.ndarray) -> np.ndarray:
        pin.forwardKinematics(self.reduced_robot.model, self.reduced_robot.data, q)
        pose = self.reduced_robot.data.oMf[self.gripper_id]
        pos = pose.translation
        euler = R.from_matrix(pose.rotation).as_euler("xyz", degrees=False)
        return np.concatenate([pos, euler], axis=0)
