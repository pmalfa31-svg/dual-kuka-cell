import numpy as np
import mujoco

class VacuumController:
    """Controllore delle ventose a depressione per i robot KUKA."""
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        self.r1_weld_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "r1_weld")
        self.r2_weld_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "r2_weld")

        self.r1_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "r1_suction_site")
        self.r2_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "r2_suction_site")
        self.box_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "box_0_site")

    def set_gripper(self, robot_id: int, activate: bool):
        weld_id = self.r1_weld_id if robot_id == 1 else self.r2_weld_id
        site_id = self.r1_site_id if robot_id == 1 else self.r2_site_id

        if activate:
            # Tolleranza di prossimita a 11 cm per aggancio dinamico affidabile
            dist = np.linalg.norm(self.data.site_xpos[site_id] - self.data.site_xpos[self.box_site_id])
            if dist < 0.11:
                self.data.eq_active[weld_id] = 1
        else:
            self.data.eq_active[weld_id] = 0

    def is_holding(self, robot_id: int) -> bool:
        weld_id = self.r1_weld_id if robot_id == 1 else self.r2_weld_id
        return bool(self.data.eq_active[weld_id])
