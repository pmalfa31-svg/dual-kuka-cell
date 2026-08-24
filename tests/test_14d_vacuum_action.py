import numpy as np
import mujoco
from src.envs.dual_arm_env import DualKukaPalletizeEnv
from src.core.expert import IndustrialExpertPlanner

def test_action_space_is_14d_and_expert_matches_it():
    env = DualKukaPalletizeEnv()
    expert = IndustrialExpertPlanner(env.pallet_1, env.pallet_2)
    obs, _ = env.reset(seed=1)
    action = expert.get_expert_action(env)

    assert env.action_space.shape == (14,)
    assert action.shape == (14,)
    assert np.all(action >= -1.0) and np.all(action <= 1.0)

def test_vacuum_is_caused_by_action_and_can_release():
    env = DualKukaPalletizeEnv()
    env.reset(seed=2)

    box_jnt_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "box_0_joint")
    box_qpos_adr = env.model.jnt_qposadr[box_jnt_id]

    # Posiziona il collo direttamente sotto il TCP del Robot 1 (Nord)
    p_ee1, _ = env.kinematics.get_ee_positions()
    env.data.qpos[box_qpos_adr:box_qpos_adr+3] = [p_ee1[0], p_ee1[1], p_ee1[2] - 0.05]
    mujoco.mj_forward(env.model, env.data)

    # 1. Azione con vuoto spento (g1 = -1.0) -> Non deve agganciare
    action_off = np.zeros(14, dtype=np.float32)
    action_off[6] = -1.0
    action_off[13] = -1.0
    env.step(action_off)
    assert not env.vacuum.is_holding(1)

    # 2. Azione con vuoto acceso (g1 = 1.0) -> Deve agganciare
    action_on = np.zeros(14, dtype=np.float32)
    action_on[6] = 1.0
    action_on[13] = -1.0
    env.step(action_on)
    assert env.vacuum.is_holding(1)

    # 3. Azione di rilascio (g1 = -1.0) -> Deve sganciare
    env.step(action_off)
    assert not env.vacuum.is_holding(1)
