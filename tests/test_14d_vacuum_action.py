import numpy as np

from src.envs.dual_arm_env import DualKukaPalletizeEnv
from src.core.expert import IndustrialExpertPlanner


def test_action_space_is_14d_and_expert_matches_it():
    env = DualKukaPalletizeEnv()
    env.reset(seed=1)

    assert env.action_space.shape == (14,)

    expert = IndustrialExpertPlanner(env.pallet_1, env.pallet_2)
    action = expert.get_expert_action(env)
    assert action.shape == (14,)
    assert np.all(action >= -1.0)
    assert np.all(action <= 1.0)


def test_vacuum_is_caused_by_action_and_can_release():
    env = DualKukaPalletizeEnv()
    env.reset(seed=2)

    box_joint = env.model.jnt_qposadr[
        env.model.name2id("box_0_joint", "joint")
    ]
    ee = env.kinematics.get_ee_positions()[0].copy()
    env.data.qpos[box_joint:box_joint + 3] = ee + np.array([0.0, 0.0, -0.04])
    env.data.qpos[box_joint + 3:box_joint + 7] = [1.0, 0.0, 0.0, 0.0]

    import mujoco
    mujoco.mj_forward(env.model, env.data)

    # Robot 1: zero joint motion + vacuum ON.
    action_on = np.zeros(14, dtype=np.float32)
    action_on[6] = 1.0
    _, _, _, _, info = env.step(action_on)
    assert info["holding_r1"] is True

    # Robot 1: vacuum OFF must release the weld.
    action_off = np.zeros(14, dtype=np.float32)
    action_off[6] = -1.0
    _, _, _, _, info = env.step(action_off)
    assert info["holding_r1"] is False
