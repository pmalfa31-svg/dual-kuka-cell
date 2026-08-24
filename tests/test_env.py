import numpy as np
from src.envs.dual_arm_env import DualKukaPalletizeEnv

def test_env_lifecycle():
    env = DualKukaPalletizeEnv()
    obs, info = env.reset()
    
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (64,)
    
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    assert obs.shape == (64,)
    assert isinstance(reward, (float, np.floating))
    assert isinstance(terminated, (bool, np.bool_))
    assert isinstance(truncated, (bool, np.bool_))
    assert "is_success" in info
    assert "collision" in info
