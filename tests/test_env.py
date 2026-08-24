import pytest
import numpy as np
from gymnasium.utils.env_checker import check_env
from src.envs.dual_arm_env import DualKukaPalletizeEnv

def test_gym_compliance():
    env = DualKukaPalletizeEnv()
    # Validate Gymnasium API standards
    check_env(env.unwrapped, skip_render_check=True)

def test_step_and_reset():
    env = DualKukaPalletizeEnv()
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (64,)
    
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == (64,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "is_success" in info
    assert "collision" in info
