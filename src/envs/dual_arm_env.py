import os
import yaml
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

from src.core.conveyor import OvalConveyor
from src.core.kinematics import CellKinematics
from src.controllers.vacuum import VacuumController

class DualKukaPalletizeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, config_path: str = "configs/training_cfg.yaml", render_mode: str = None):
        super().__init__()
        
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.model = mujoco.MjModel.from_xml_path(self.cfg["env"]["xml_path"])
        self.data = mujoco.MjData(self.model)

        self.frame_skip = self.cfg["env"]["frame_skip"]
        self.max_steps = self.cfg["env"]["max_episode_steps"]
        self.max_delta_q = self.cfg["env"]["max_delta_q"]
        self.current_step = 0

        self.conveyor = OvalConveyor(speed=self.cfg["env"]["conveyor_speed"])
        self.kinematics = CellKinematics(self.model, self.data)
        self.vacuum = VacuumController(self.model, self.data)

        self.q_home = np.array([0.0, -1.2, 1.8, 0.0, -0.6, 0.0], dtype=np.float32)
        self.q_target_1 = self.q_home.copy()
        self.q_target_2 = self.q_home.copy()

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(14,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(64,), dtype=np.float32)

        self.pallet_1 = np.array(self.cfg["pallet_targets"]["robot_1"], dtype=np.float32)
        self.pallet_2 = np.array(self.cfg["pallet_targets"]["robot_2"], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.current_step = 0

        self.data.qpos[0:6] = self.q_home
        self.data.qpos[6:12] = self.q_home
        self.q_target_1 = self.q_home.copy()
        self.q_target_2 = self.q_home.copy()

        box_qpos_adr = self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "box_0_joint")]
        noise_x = np.random.uniform(-0.03, 0.03)
        self.data.qpos[box_qpos_adr:box_qpos_adr+3] = [2.78 + noise_x, 0.0, 0.73]
        self.data.qpos[box_qpos_adr+3:box_qpos_adr+7] = [0.989, 0.0, -0.149, 0.0]

        self.data.ctrl[0:6] = self.q_home
        self.data.ctrl[6:12] = self.q_home
        self.vacuum.set_gripper(1, False)
        self.vacuum.set_gripper(2, False)

        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        self.current_step += 1
        action = np.clip(action, -1.0, 1.0)

        a_r1 = action[0:7]
        a_r2 = action[7:14]

        self.q_target_1 = np.clip(self.q_target_1 + a_r1[0:6] * self.max_delta_q,
                                  self.model.jnt_range[0:6, 0], self.model.jnt_range[0:6, 1])
        self.q_target_2 = np.clip(self.q_target_2 + a_r2[0:6] * self.max_delta_q,
                                  self.model.jnt_range[6:12, 0], self.model.jnt_range[6:12, 1])

        self.data.ctrl[0:6] = self.q_target_1
        self.data.ctrl[6:12] = self.q_target_2

        self.vacuum.set_gripper(1, activate=(a_r1[6] > 0.0))
        self.vacuum.set_gripper(2, activate=(a_r2[6] > 0.0))

        for _ in range(self.frame_skip):
            self.conveyor.apply_conveyor_velocity(self.model, self.data, "box_0")
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward_1, reward_2 = self._compute_rewards(a_r1, a_r2)
        total_reward = reward_1 + reward_2

        collision = self.kinematics.check_robot_collision()
        box_pos, _ = self.kinematics.get_box_pose()

        succ_1 = (np.linalg.norm(box_pos[0:2] - self.pallet_1[0:2]) < 0.3) and (box_pos[2] < 0.28) and not self.vacuum.is_holding(1)
        succ_2 = (np.linalg.norm(box_pos[0:2] - self.pallet_2[0:2]) < 0.3) and (box_pos[2] < 0.28) and not self.vacuum.is_holding(2)
        success = succ_1 or succ_2

        terminated = collision or success
        truncated = self.current_step >= self.max_steps

        info = {
            "reward_r1": reward_1,
            "reward_r2": reward_2,
            "is_success": success,
            "collision": collision,
            "holding_r1": self.vacuum.is_holding(1),
            "holding_r2": self.vacuum.is_holding(2)
        }

        return obs, total_reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        q_r1, q_r2 = self.data.qpos[0:6], self.data.qpos[6:12]
        dq_r1, dq_r2 = self.data.qvel[0:6], self.data.qvel[6:12]
        p_ee1, p_ee2 = self.kinematics.get_ee_positions()
        box_pos, box_vel = self.kinematics.get_box_pose()

        dist_ee = np.linalg.norm(p_ee1 - p_ee2)

        obs_1 = np.concatenate([
            q_r1, dq_r1, p_ee1,
            box_pos, box_vel,
            box_pos - p_ee1,
            self.pallet_1 - box_pos,
            [float(self.vacuum.is_holding(1))],
            p_ee2, [dist_ee]
        ], dtype=np.float32)

        obs_2 = np.concatenate([
            q_r2, dq_r2, p_ee2,
            box_pos, box_vel,
            box_pos - p_ee2,
            self.pallet_2 - box_pos,
            [float(self.vacuum.is_holding(2))],
            p_ee1, [dist_ee]
        ], dtype=np.float32)

        return np.concatenate([obs_1, obs_2], dtype=np.float32)

    def _compute_rewards(self, a_r1: np.ndarray, a_r2: np.ndarray) -> tuple[float, float]:
        cfg_r = self.cfg["reward_weights"]
        p_ee1, p_ee2 = self.kinematics.get_ee_positions()
        box_pos, _ = self.kinematics.get_box_pose()

        d_reach_1 = np.linalg.norm(p_ee1 - box_pos)
        d_reach_2 = np.linalg.norm(p_ee2 - box_pos)

        r1 = -cfg_r["w_reach"] * d_reach_1 - cfg_r["p_energy"] * np.sum(np.square(a_r1))
        r2 = -cfg_r["w_reach"] * d_reach_2 - cfg_r["p_energy"] * np.sum(np.square(a_r2))

        # Guida alla presa: bonus se attiva la ventosa quando si trova vicino al collo (< 15 cm)
        if a_r1[6] > 0.0 and d_reach_1 < 0.15:
            r1 += 2.0
        if a_r2[6] > 0.0 and d_reach_2 < 0.15:
            r2 += 2.0

        if self.vacuum.is_holding(1):
            r1 += cfg_r["w_grasp"]
            r1 -= cfg_r["w_transport"] * np.linalg.norm(box_pos - self.pallet_1)

        if self.vacuum.is_holding(2):
            r2 += cfg_r["w_grasp"]
            r2 -= cfg_r["w_transport"] * np.linalg.norm(box_pos - self.pallet_2)

        if self.kinematics.check_robot_collision():
            r1 -= cfg_r["p_collision"]
            r2 -= cfg_r["p_collision"]

        return float(r1), float(r2)
