import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from src.envs.dual_arm_env import DualKukaPalletizeEnv

def make_env(rank: int, seed: int = 0):
    def _init():
        env = DualKukaPalletizeEnv()
        env.reset(seed=seed + rank)
        return Monitor(env)
    return _init

def main():
    log_dir = "logs/tensorboard/"
    checkpoint_dir = "checkpoints/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 4 ambienti paralleli per saturare i core CPU
    num_cpu = 4
    print(f"[*] Inizializzazione di {num_cpu} ambienti paralleli su CPU...")
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    eval_env = SubprocVecEnv([make_env(999)])

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=checkpoint_dir,
        log_path=log_dir,
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=checkpoint_dir,
        name_prefix="dual_kuka_ppo"
    )

    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=torch.nn.ReLU
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=log_dir
    )

    print("[*] Avvio Addestramento PPO Multi-Agent Dual-KUKA (CPU-Vectorized)...")
    total_timesteps = 500_000
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )

    model.save(os.path.join(checkpoint_dir, "dual_kuka_final_model"))
    print(f"[+] Addestramento completato. Modello salvato in {checkpoint_dir}")

if __name__ == "__main__":
    main()
