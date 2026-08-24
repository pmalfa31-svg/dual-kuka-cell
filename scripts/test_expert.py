import time
import numpy as np
import mujoco
import mujoco.viewer

from src.envs.dual_arm_env import DualKukaPalletizeEnv
from src.core.expert import IndustrialExpertPlanner

def main():
    env = DualKukaPalletizeEnv()
    expert = IndustrialExpertPlanner(env.pallet_1, env.pallet_2)

    obs, _ = env.reset()
    print("\n" + "=" * 70)
    print("      DIAGNOSTICA ESPERTO - VALIDAZIONE TRAIETTORIA NOMINALE      ")
    print("=" * 70)

    step_count = 0
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running() and step_count < env.max_steps:
            t0 = time.time()
            action = expert.get_expert_action(env)
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1

            p_ee1, p_ee2 = env.kinematics.get_ee_positions()
            p_box, _ = env.kinematics.get_box_pose()

            if step_count % 20 == 0 or info["holding_r1"] or info["holding_r2"] or info["is_success"]:
                print(f"[Step {step_count:4d}] "
                      f"Box: [{p_box[0]:.2f}, {p_box[1]:.2f}, {p_box[2]:.2f}] | "
                      f"EE1: [{p_ee1[0]:.2f}, {p_ee1[1]:.2f}, {p_ee1[2]:.2f}] | "
                      f"EE2: [{p_ee2[0]:.2f}, {p_ee2[1]:.2f}, {p_ee2[2]:.2f}] | "
                      f"Hold1: {str(info['holding_r1']):<5} | Hold2: {str(info['holding_r2']):<5} | "
                      f"Success: {info['is_success']}")

            viewer.sync()
            time_until_next_frame = (1.0 / 60.0) - (time.time() - t0)
            if time_until_next_frame > 0:
                time.sleep(time_until_next_frame)

            if info["is_success"]:
                print(f"\n[+] SUCCESSO CONFERMATO al step {step_count} (Tempo ciclo: {step_count * 0.01:.2f} s)!")
                break
            if terminated or truncated:
                print(f"\n[-] Terminato (Collisione: {info['collision']}, Timeout: {truncated})")
                break

if __name__ == "__main__":
    main()
