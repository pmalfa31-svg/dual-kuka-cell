import numpy as np

class IndustrialExpertPlanner:
    """Pianificatore a stati finiti con finestre operative di presa e svincolo in quota."""
    def __init__(self, pallet_1: np.ndarray, pallet_2: np.ndarray, max_delta: float = 0.04):
        self.pallet_1 = pallet_1
        self.pallet_2 = pallet_2
        self.max_delta = max_delta

    def get_expert_action(self, env) -> np.ndarray:
        p_ee1, p_ee2 = env.kinematics.get_ee_positions()
        p_box, v_box = env.kinematics.get_box_pose()

        # ==================== ROBOT 1 (NORD) ====================
        if env.vacuum.is_holding(1):
            if p_ee1[2] < 0.60 and np.linalg.norm(p_ee1[0:2] - self.pallet_1[0:2]) > 0.4:
                # Svincolo verticale di sicurezza prima della traslazione
                target_1 = np.array([p_ee1[0], p_ee1[1], 0.65], dtype=np.float32)
            else:
                # Trasferimento sul pallet e discesa di scarico
                target_1 = np.array([self.pallet_1[0], self.pallet_1[1], 0.32], dtype=np.float32)
        elif -0.50 <= p_box[0] <= 0.60 and p_box[1] > 0.20:
            # Finestra di cattura attiva sul nastro Nord: intercetta con anticipo dinamico
            target_1 = np.array([p_box[0] - 0.03, p_box[1], p_box[2] + 0.04], dtype=np.float32)
        else:
            # Pre-Pick Standby sopra il nastro Nord
            target_1 = np.array([0.15, 0.60, 0.62], dtype=np.float32)

        diff_1 = target_1 - p_ee1
        a_r1 = np.clip(diff_1 / self.max_delta, -1.0, 1.0)

        # ==================== ROBOT 2 (SUD) ====================
        if env.vacuum.is_holding(2):
            if p_ee2[2] < 0.60 and np.linalg.norm(p_ee2[0:2] - self.pallet_2[0:2]) > 0.4:
                target_2 = np.array([p_ee2[0], p_ee2[1], 0.65], dtype=np.float32)
            else:
                target_2 = np.array([self.pallet_2[0], self.pallet_2[1], 0.32], dtype=np.float32)
        elif -0.60 <= p_box[0] <= 0.50 and p_box[1] < -0.20:
            # Finestra di cattura attiva sul nastro Sud
            target_2 = np.array([p_box[0] + 0.03, p_box[1], p_box[2] + 0.04], dtype=np.float32)
        else:
            # Pre-Pick Standby sopra il nastro Sud
            target_2 = np.array([-0.15, -0.60, 0.62], dtype=np.float32)

        diff_2 = target_2 - p_ee2
        a_r2 = np.clip(diff_2 / self.max_delta, -1.0, 1.0)

        return np.concatenate([a_r1, a_r2], dtype=np.float32)
