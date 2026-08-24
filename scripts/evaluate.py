import time
import numpy as np
from src.envs.dual_arm_env import DualKukaPalletizeEnv

def main():
    env = DualKukaPalletizeEnv()
    print("=== AVVIO STRESS TEST GYMNASIUM DUAL-KUKA CELL ===")
    print(f"Observation Space: {env.observation_space}")
    print(f"Action Space:      {env.action_space}")

    num_episodes = 5
    t0 = time.time()
    total_steps = 0

    for ep in range(1, num_episodes + 1):
        obs, _ = env.reset()
        ep_reward = 0.0
        steps = 0

        while True:
            # Azioni casuali uniformi
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            steps += 1

            if terminated or truncated:
                break

        total_steps += steps
        print(f"Episode {ep:02d} | Steps: {steps:03d} | Total Reward: {ep_reward:8.2f} | "
              f"Collision: {info['collision']} | Success: {info['is_success']}")

    elapsed = time.time() - t0
    fps = total_steps / elapsed
    print("==================================================")
    print(f"Test Completato: {total_steps} passi in {elapsed:.2f}s (~{fps:.0f} FPS su CPU)")

if __name__ == "__main__":
    main()
