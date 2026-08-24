import numpy as np
import mujoco

class CellKinematics:
    """Utility per l'estrazione dello stato geometrico e verifica collisioni."""
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        self.r1_ee_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "r1_suction_site")
        self.r2_ee_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "r2_suction_site")
        self.box_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "box_0_site")

    def get_ee_positions(self) -> tuple[np.ndarray, np.ndarray]:
        """Restituisce le coordinate cartesiane (X, Y, Z) degli End-Effector."""
        p1 = self.data.site_xpos[self.r1_ee_site].copy()
        p2 = self.data.site_xpos[self.r2_ee_site].copy()
        return p1, p2

    def get_box_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Restituisce posizione lineare (3) e velocita lineare (3) del collo."""
        box_pos = self.data.site_xpos[self.box_site].copy()
        box_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "box_0")
        box_vel = self.data.cvel[box_id][3:6].copy()  # [vx, vy, vz]
        return box_pos, box_vel

    def check_robot_collision(self) -> bool:
        """Rileva collisioni strutturali tra le geometrie del Robot 1 e Robot 2."""
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
            geom2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
            
            if geom1_name and geom2_name:
                # Se entrambi i geom appartengono a robot diversi
                if ("r1_" in geom1_name and "r2_" in geom2_name) or ("r2_" in geom1_name and "r1_" in geom2_name):
                    return True
        return False
