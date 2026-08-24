import numpy as np
import mujoco

class VacuumController:
    """Controllore pneumatico industriale ad azione esplicita (g > 0 = ON, g <= 0 = OFF)."""
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        self.r1_weld_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "r1_weld")
        self.r2_weld_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "r2_weld")
        self.r1_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "r1_suction_site")
        self.r2_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "r2_suction_site")
        self.box_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box_0")

    def command(self, robot_id: int, vacuum_cmd: float):
        weld_id = self.r1_weld_id if robot_id == 1 else self.r2_weld_id
        site_id = self.r1_site_id if robot_id == 1 else self.r2_site_id
        other_weld_id = self.r2_weld_id if robot_id == 1 else self.r1_weld_id

        if vacuum_cmd > 0.0:
            p_box = self.data.xpos[self.box_body_id]
            p_ee = self.data.site_xpos[site_id]
            dist_3d = np.linalg.norm(p_ee - p_box)
            if dist_3d < 0.25 and self.data.eq_active[other_weld_id] == 0:
                self.data.eq_active[weld_id] = 1
        else:
            self.data.eq_active[weld_id] = 0

    def apply_commands(self, g1: float, g2: float):
        self.command(1, g1)
        self.command(2, g2)

    def apply_action(self, g1: float, g2: float):
        self.apply_commands(g1, g2)

    def release_all(self):
        self.data.eq_active[self.r1_weld_id] = 0
        self.data.eq_active[self.r2_weld_id] = 0

    def is_holding(self, robot_id: int) -> bool:
        weld_id = self.r1_weld_id if robot_id == 1 else self.r2_weld_id
        return bool(self.data.eq_active[weld_id])
