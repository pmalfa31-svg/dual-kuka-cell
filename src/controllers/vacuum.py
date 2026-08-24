import numpy as np
import mujoco

class VacuumController:
    """Controllore industriale: aggancio su contatto e rilascio immediato sul pallet."""
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        self.r1_weld_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "r1_weld")
        self.r2_weld_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "r2_weld")

        self.r1_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "r1_suction_site")
        self.r2_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "r2_suction_site")
        self.box_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box_0")

    def step_auto(self, pallet_1: np.ndarray, pallet_2: np.ndarray):
        p_box = self.data.xpos[self.box_body_id]
        p_ee1 = self.data.site_xpos[self.r1_site_id]
        p_ee2 = self.data.site_xpos[self.r2_site_id]

        # Robot 1 (North)
        if self.data.eq_active[self.r1_weld_id] == 1:
            # Release garantito appena sopra il pallet
            if np.linalg.norm(p_box[0:2] - pallet_1[0:2]) < 0.35 and p_box[2] < 0.45:
                self.data.eq_active[self.r1_weld_id] = 0
        else:
            d_xy = np.linalg.norm(p_ee1[0:2] - p_box[0:2])
            d_z = abs(p_ee1[2] - (p_box[2] + 0.04))
            if d_xy < 0.14 and d_z < 0.09 and self.data.eq_active[self.r2_weld_id] == 0:
                self.data.eq_active[self.r1_weld_id] = 1

        # Robot 2 (South)
        if self.data.eq_active[self.r2_weld_id] == 1:
            if np.linalg.norm(p_box[0:2] - pallet_2[0:2]) < 0.35 and p_box[2] < 0.45:
                self.data.eq_active[self.r2_weld_id] = 0
        else:
            d_xy = np.linalg.norm(p_ee2[0:2] - p_box[0:2])
            d_z = abs(p_ee2[2] - (p_box[2] + 0.04))
            if d_xy < 0.14 and d_z < 0.09 and self.data.eq_active[self.r1_weld_id] == 0:
                self.data.eq_active[self.r2_weld_id] = 1

    def release_all(self):
        self.data.eq_active[self.r1_weld_id] = 0
        self.data.eq_active[self.r2_weld_id] = 0

    def is_holding(self, robot_id: int) -> bool:
        weld_id = self.r1_weld_id if robot_id == 1 else self.r2_weld_id
        return bool(self.data.eq_active[weld_id])
