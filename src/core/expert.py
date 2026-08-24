import numpy as np

class IndustrialExpertPlanner:
    """Pianificatore Traiettorie in Joint Space."""
    def __init__(self, pallet_1: np.ndarray, pallet_2: np.ndarray):
        # Carica le configurazioni esatte dei giunti calcolate offline
        data = np.load("data/calibrated_waypoints.npz")
        self.wp_r1 = data["wp_r1"]  # [HOME, PREPICK, PICK, LIFT, PLACE]
        self.wp_r2 = data["wp_r2"]
        self.reset()

    def reset(self):
        self.phase_r1 = 0
        self.phase_r2 = 0
        self.g1 = -1.0
        self.g2 = -1.0
        self.active_robot = 1

    def get_expert_action(self, env) -> np.ndarray:
        q_r1 = env.data.qpos[0:6]
        q_r2 = env.data.qpos[6:12]
        p_box, _ = env.kinematics.get_box_pose()

        # ======================= ROBOT 1 (NORD) =======================
        target_q1 = self.wp_r1[0]  # Default Home

        if self.active_robot == 1:
            target_q1 = self.wp_r1[self.phase_r1]
            dist_jnt = np.linalg.norm(target_q1 - q_r1)

            if self.phase_r1 == 0:  # Attesa a HOME
                if abs(p_box[0]) < 0.15 and p_box[1] > 0.30:
                    self.phase_r1 = 1  # Trigger discesa a PREPICK

            elif self.phase_r1 == 1:  # PREPICK
                if dist_jnt < 0.15:
                    self.phase_r1 = 2  # Passa a PICK
                    self.g1 = 1.0      # Attiva ventosa preventivamente

            elif self.phase_r1 == 2:  # PICK
                self.g1 = 1.0
                if env.vacuum.is_holding(1):
                    self.phase_r1 = 3  # Agganciato, passa a LIFT

            elif self.phase_r1 == 3:  # LIFT
                self.g1 = 1.0
                if dist_jnt < 0.15:
                    self.phase_r1 = 4  # Quota raggiunta, passa a PLACE

            elif self.phase_r1 == 4:  # PLACE
                self.g1 = 1.0
                if dist_jnt < 0.15:    # Siamo sopra il pallet
                    self.g1 = -1.0     # Sgancio
                    self.phase_r1 = 0  # Reset per il prossimo ciclo

        # ======================= AZIONE =======================
        # Satura sempre l'output a max velocità [-1, 1]
        dq1 = np.clip((target_q1 - q_r1) / 0.05, -1.0, 1.0)
        
        # Manteniamo il Robot 2 fisso in Home per test
        dq2 = np.clip((self.wp_r2[0] - q_r2) / 0.05, -1.0, 1.0)

        act_r1 = np.append(dq1, np.float32(self.g1))
        act_r2 = np.append(dq2, np.float32(-1.0))
        return np.concatenate([act_r1, act_r2], dtype=np.float32)
