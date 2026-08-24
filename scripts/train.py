import os
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from src.envs.dual_arm_env import DualKukaPalletizeEnv

def make_env():
    def _init():
        return DualKukaPalletizeEnv()
    return _init

def main():
    with open("configs/training_cfg.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs/tensorboard", exist_ok=True)

    num_envs = 4
    vec_env = SubprocVecEnv([make_env() for _ in range(num_envs)])

    model = PPO(
        "MlpPolicy",
        vec_env,
        n_steps=1024,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        learning_rate=3e-4,
        policy_kwargs={"net_arch": [256, 256]},
        tensorboard_log="logs/tensorboard/",
        verbose=1
    )

    eval_env = DualKukaPalletizeEnv()
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="checkpoints/",
        log_path="logs/",
        eval_freq=5000,
        deterministic=True,
        render=False
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=25000,
        save_path="checkpoints/",
        name_prefix="kuka_ppo"
    )

    total_timesteps = 500000
    print(f"[*] Avvio training PPO Cartesiano con Progress Bar ({total_timesteps} timesteps)...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )

    model.save("checkpoints/final_model")
    print("[+] Addestramento completato.")

if __name__ == "__main__":
    main()
