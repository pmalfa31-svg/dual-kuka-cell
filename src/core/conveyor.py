import numpy as np
import mujoco

class OvalConveyor:
    """
    Controllore cinematico a campo vettoriale continuo per nastro ovale chiuso.
    Garantisce continuita C1 della velocita tangenziale e correzione radiale coerente.
    """
    def __init__(self, speed: float = 0.6):
        self.speed = speed
        self.straight_len = 1.2
        self.track_radius = 0.6
        self.belt_height = 0.45
        self.kp_track = 3.0  # Guadagno proporzionale di richiamo

    def apply_conveyor_velocity(self, model: mujoco.MjModel, data: mujoco.MjData, box_body_name: str = "box_0"):
        box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, box_body_name)
        box_pos = data.xpos[box_id]

        # Se il pacco e sollevato dalla ventosa o fuori dal piano, non applicare trazione
        if abs(box_pos[2] - self.belt_height) > 0.15:
            return

        x, y = box_pos[0], box_pos[1]

        # 1. CURVA OVEST (x < -1.2)
        if x < -self.straight_len:
            cx, cy = -self.straight_len, 0.0
            dx, dy = x - cx, y - cy
            r = np.sqrt(dx**2 + dy**2)
            if r > 0.001:
                # Vettore tangente unitario orario: (-dy/r, dx/r)
                tx, ty = -dy / r, dx / r
                # Errore rispetto al raggio nominale (punto target sulla linea di mezzeria)
                err_x = (cx + self.track_radius * (dx / r)) - x
                err_y = (cy + self.track_radius * (dy / r)) - y

                vx = self.speed * tx + self.kp_track * err_x
                vy = self.speed * ty + self.kp_track * err_y
            else:
                vx, vy = 0.0, -self.speed

        # 2. CURVA EST (x > +1.2)
        elif x > self.straight_len:
            cx, cy = self.straight_len, 0.0
            dx, dy = x - cx, y - cy
            r = np.sqrt(dx**2 + dy**2)
            if r > 0.001:
                # Vettore tangente unitario orario: (-dy/r, dx/r)
                tx, ty = -dy / r, dx / r
                # Errore rispetto al raggio nominale
                err_x = (cx + self.track_radius * (dx / r)) - x
                err_y = (cy + self.track_radius * (dy / r)) - y

                vx = self.speed * tx + self.kp_track * err_x
                vy = self.speed * ty + self.kp_track * err_y
            else:
                vx, vy = 0.0, self.speed

        # 3. TRATTI RETTILINEI (Nord e Sud)
        else:
            if y >= 0.0:
                # Tratto Nord: viaggia verso -X, agganciato a Y = +0.6
                vx = -self.speed
                vy = self.kp_track * (self.track_radius - y)
            else:
                # Tratto Sud: viaggia verso +X, agganciato a Y = -0.6
                vx = self.speed
                vy = self.kp_track * (-self.track_radius - y)

        # Assegnazione al joint libero del pacco
        jnt_qvel_addr = model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "box_0_joint")]
        data.qvel[jnt_qvel_addr:jnt_qvel_addr+2] = [vx, vy]
