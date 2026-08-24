import time
import argparse
import numpy as np
import mujoco.viewer
from stable_baselines3 import PPO
from src.envs.dual_arm_env import DualKukaPalletizeEnv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="checkpoints/best_model.zip", help="Path del file .zip del modello")
    args = parser.parse_args()

    print(f"[*] Caricamento modello da: {args.model}")
    env = DualKukaPalletizeEnv()
    model = PPO.load(args.model)

    obs, _ = env.reset()
    
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # Inferenza deterministica della Policy
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            viewer.sync()

            if terminated or truncated:
                print(f"[EPISODE END] Success: {info['is_success']} | Collision: {info['collision']}")
                obs, _ = env.reset()

            # Mantenimento real-time sync
            dt_step = env.model.opt.timestep * env.frame_skip
            time_until_next_step = dt_step - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
