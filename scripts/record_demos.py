import os
import sys
import numpy as np
import mujoco
from tqdm import tqdm
from src.envs.dual_arm_env import DualKukaPalletizeEnv
from src.core.expert import IndustrialExpertPlanner

def main():
    env = DualKukaPalletizeEnv()
    expert = IndustrialExpertPlanner(env.pallet_1, env.pallet_2)

    observations = []
    actions = []
    target_episodes = 50

    pbar = tqdm(total=target_episodes, desc="[Record 14D] Raccolta dataset esperto", unit="ep")

    successful_episodes = 0
    box_jnt_adr = env.model.jnt_qposadr[mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "box_0_joint")]

    while successful_episodes < target_episodes:
        obs, _ = env.reset()
        expert.reset()
        
        # Testiamo solo il Robot 1 (Nord) per isolare il problema
        expert.active_robot = 1 
        
        # Iniezione del pacco esattamente sotto la ventosa
        env.data.qpos[box_jnt_adr:box_jnt_adr+3] = [0.0, 0.60, 0.46]
        env.data.qpos[box_jnt_adr+3:box_jnt_adr+7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(env.model, env.data)

        ep_obs, ep_act = [], []
        
        for step_i in range(1, 350):
            action = expert.get_expert_action(env)
            ep_obs.append(obs)
            ep_act.append(action)

            obs, reward, terminated, truncated, info = env.step(action)

            if info["is_success"]:
                successful_episodes += 1
                observations.extend(ep_obs)
                actions.extend(ep_act)
                pbar.update(1)
                pbar.set_postfix({"Transizioni": len(observations)})
                break
            
            if terminated or truncated or step_i == 349:
                pbar.close()
                p_box, _ = env.kinematics.get_box_pose()
                d_pallet = np.linalg.norm(p_box[0:2] - env.pallet_1[0:2])
                
                print("\n" + "="*60)
                print(" FATAL ERROR: L'EPISODIO HA FALLITO E IL CICLO SI E' FERMATO")
                print("="*60)
                print(f"Frame finale raggiunto : {step_i}")
                print(f"Stato Expert R1        : {getattr(expert, 'phase_r1', 'N/A')}")
                print(f"Stato Ventosa (Hold1)  : {info['holding_r1']}")
                print(f"Distanza dal Pallet    : {d_pallet:.3f} m (Soglia: < 0.38 m)")
                print(f"Quota Z del Pacco      : {p_box[2]:.3f} m (Soglia: < 0.48 m)")
                print(f"Collisione Rilevata    : {info['collision']}")
                print("="*60)
                sys.exit(1) # Blocca lo script istantaneamente

    pbar.close()
    os.makedirs("data", exist_ok=True)
    np.savez_compressed(
        "data/expert_dataset.npz",
        obs=np.array(observations, dtype=np.float32),
        actions=np.array(actions, dtype=np.float32)
    )
    print(f"\n[+] Dataset 14D salvato: {len(observations)} transizioni.")

if __name__ == "__main__":
    main()
