import numpy as np
import mujoco

class CellKinematics:
    """Cinematica cellulare con risolutore IK a convergenza completa su MjData isolato."""
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        self.ik_data = mujoco.MjData(model)

        self.r1_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "r1_suction_site")
        self.r2_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "r2_suction_site")
        self.box_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box_0")

    def get_ee_positions(self) -> tuple[np.ndarray, np.ndarray]:
        return self.data.site_xpos[self.r1_site].copy(), self.data.site_xpos[self.r2_site].copy()

    def get_box_pose(self) -> tuple[np.ndarray, np.ndarray]:
        pos = self.data.xpos[self.box_body].copy()
        vel = self.data.cvel[self.box_body][3:6].copy()
        return pos, vel

    def check_robot_collision(self, threshold: float = 0.38) -> bool:
        p1, p2 = self.get_ee_positions()
        return float(np.linalg.norm(p1 - p2)) < threshold

    def solve_ik_target(self, robot_id: int, target_xyz: np.ndarray) -> np.ndarray:
        """Risolve la cinematica inversa completa fino a target_xyz (< 1 mm di errore)."""
        site_id = self.r1_site if robot_id == 1 else self.r2_site
        q_idx = slice(0, 6) if robot_id == 1 else slice(6, 12)
        low = self.model.jnt_range[q_idx, 0]
        high = self.model.jnt_range[q_idx, 1]

        self.ik_data.qpos[:] = self.data.qpos[:]
        mujoco.mj_forward(self.model, self.ik_data)

        for _ in range(300):
            current_xyz = self.ik_data.site_xpos[site_id]
            err = target_xyz - current_xyz
            if np.linalg.norm(err) < 1e-4:
                break

            jacp = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.ik_data, jacp, None, site_id)
            J = jacp[:, q_idx]

            dq = J.T @ np.linalg.inv(J @ J.T + 1e-3 * np.eye(3)) @ err
            self.ik_data.qpos[q_idx] = np.clip(self.ik_data.qpos[q_idx] + np.clip(dq, -0.10, 0.10), low, high)
            mujoco.mj_forward(self.model, self.ik_data)

        return self.ik_data.qpos[q_idx].copy()
