import numpy as np


class IndustrialExpertPlanner:
    """Expert planner compatibile con action space [dq1..dq6,g] x 2."""

    def __init__(self, pallet_1: np.ndarray, pallet_2: np.ndarray, max_delta_q: float = 0.05):
        self.pallet_1 = pallet_1
        self.pallet_2 = pallet_2
        self.max_delta_q = max_delta_q

    def _joint_action(self, env, robot_id: int, target: np.ndarray) -> np.ndarray:
        p_ee = env.kinematics.get_ee_positions()[robot_id - 1]
        delta_xyz = target - p_ee
        dq = env.kinematics.solve_ik_delta(robot_id, delta_xyz)
        return np.clip(dq / self.max_delta_q, -1.0, 1.0).astype(np.float32)

    @staticmethod
    def _vacuum_command(env, robot_id: int, target: np.ndarray, holding: bool) -> float:
        """ON durante presa/trasporto, OFF quando siamo in posizione di deposito."""
        p_ee = env.kinematics.get_ee_positions()[robot_id - 1]

        if not holding:
            p_box, _ = env.kinematics.get_box_pose()
            d_xy = np.linalg.norm(p_ee[0:2] - p_box[0:2])
            d_z = abs(p_ee[2] - (p_box[2] + 0.04))
            return 1.0 if d_xy < 0.18 and d_z < 0.12 else -1.0

        if np.linalg.norm(p_ee[0:2] - target[0:2]) < 0.12 and p_ee[2] < 0.38:
            return -1.0
        return 1.0

    def get_expert_action(self, env) -> np.ndarray:
        p_ee1, p_ee2 = env.kinematics.get_ee_positions()
        p_box, _ = env.kinematics.get_box_pose()
        holding1 = env.vacuum.is_holding(1)
        holding2 = env.vacuum.is_holding(2)

        if holding1:
            if p_ee1[2] < 0.60 and np.linalg.norm(p_ee1[0:2] - self.pallet_1[0:2]) > 0.4:
                target_1 = np.array([p_ee1[0], p_ee1[1], 0.65], dtype=np.float32)
            else:
                target_1 = np.array([self.pallet_1[0], self.pallet_1[1], 0.32], dtype=np.float32)
        elif -0.50 <= p_box[0] <= 0.60 and p_box[1] > 0.20:
            target_1 = np.array([p_box[0] - 0.03, p_box[1], p_box[2] + 0.04], dtype=np.float32)
        else:
            target_1 = np.array([0.15, 0.60, 0.62], dtype=np.float32)

        if holding2:
            if p_ee2[2] < 0.60 and np.linalg.norm(p_ee2[0:2] - self.pallet_2[0:2]) > 0.4:
                target_2 = np.array([p_ee2[0], p_ee2[1], 0.65], dtype=np.float32)
            else:
                target_2 = np.array([self.pallet_2[0], self.pallet_2[1], 0.32], dtype=np.float32)
        elif -0.60 <= p_box[0] <= 0.50 and p_box[1] < -0.20:
            target_2 = np.array([p_box[0] + 0.03, p_box[1], p_box[2] + 0.04], dtype=np.float32)
        else:
            target_2 = np.array([-0.15, -0.60, 0.62], dtype=np.float32)

        a_r1 = self._joint_action(env, 1, target_1)
        a_r2 = self._joint_action(env, 2, target_2)
        g1 = self._vacuum_command(env, 1, self.pallet_1, holding1)
        g2 = self._vacuum_command(env, 2, self.pallet_2, holding2)

        return np.concatenate([
            a_r1, np.array([g1], dtype=np.float32),
            a_r2, np.array([g2], dtype=np.float32),
        ]).astype(np.float32)
