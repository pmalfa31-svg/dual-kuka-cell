import numpy as np
import mujoco

class CellKinematics:
    """Modulo cinematico per il calcolo delle pose, collisioni e Inverse Kinematics (IK)."""
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data

        self.r1_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "r1_suction_site")
        self.r2_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "r2_suction_site")
        self.box_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box_0")
        self.box_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "box_0_site")

    def get_ee_positions(self) -> tuple[np.ndarray, np.ndarray]:
        return self.data.site_xpos[self.r1_site].copy(), self.data.site_xpos[self.r2_site].copy()

    def get_box_pose(self) -> tuple[np.ndarray, np.ndarray]:
        pos = self.data.xpos[self.box_body].copy()
        vel = self.data.cvel[self.box_body][3:6].copy()
        return pos, vel

    def check_robot_collision(self, threshold: float = 0.40) -> bool:
        p1, p2 = self.get_ee_positions()
        return float(np.linalg.norm(p1 - p2)) < threshold

    def solve_ik_delta(self, robot_id: int, delta_xyz: np.ndarray, damping: float = 0.05) -> np.ndarray:
        """Risolutore Damped Least Squares (DLS) IK per il controllo cartesiano del TCP."""
        site_id = self.r1_site if robot_id == 1 else self.r2_site
        qpos_idx = slice(0, 6) if robot_id == 1 else slice(6, 12)

        # Calcola Jacobiano di traslazione (3xNV)
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, None, site_id)
        J = jacp[:, qpos_idx]  # 3x6

        # DLS: delta_q = J^T * (J * J^T + lambda^2 * I)^(-1) * delta_xyz
        lambda_eye = (damping ** 2) * np.eye(3)
        inv_term = np.linalg.inv(J @ J.T + lambda_eye)
        delta_q = J.T @ (inv_term @ delta_xyz)
        return delta_q
