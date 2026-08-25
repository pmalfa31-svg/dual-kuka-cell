import os
import numpy as np
import robosuite as suite
from tqdm import tqdm

def main():
    print("[*] Avvio registrazione dimostrazioni bimanuali...")
    
    env = suite.make(
        env_name="TwoArmLift",
        robots=["IIWA", "IIWA"],
        has_renderer=False,          # Headless per massima velocita
        has_offscreen_renderer=False,
        control_freq=20,
        horizon=100,
        use_camera_obs=False
    )

    observations = []
    actions = []
    num_episodes = 50

    pbar = tqdm(total=num_episodes, desc="[Dataset] Registrazione episodi", unit="ep")

    for ep in range(num_episodes):
        obs = env.reset()
        ep_obs, ep_act = [], []
        
        # Estrazione stato compatto (pose giunti, posizioni EE, posizione oggetto)
        flat_obs = np.concatenate([
            obs["robot0_proprio-state"],
            obs["robot1_proprio-state"],
            obs["object-state"]
        ], dtype=np.float32)

        for step in range(100):
            action = np.zeros(env.action_dim, dtype=np.float32)
            
            # FASE 1: Avvicinamento alle maniglie dell'oggetto (step 0-30)
            if step < 30:
                action[0] = 0.25   # Robot 0 avanza in X
                action[2] = -0.30  # Robot 0 scende in Z
                action[6] = -1.0   # Gripper 0 aperto
                
                action[7] = -0.25  # Robot 1 avanza in X
                action[9] = -0.30  # Robot 1 scende in Z
                action[13] = -1.0  # Gripper 1 aperto

            # FASE 2: Chiusura pinze (step 30-45)
            elif step < 45:
                action[6] = 1.0    # Gripper 0 chiuso
                action[13] = 1.0   # Gripper 1 chiuso

            # FASE 3: Sollevamento coordinato (step 45-100)
            else:
                action[2] = 0.40   # Robot 0 solleva in Z
                action[6] = 1.0    # Mantiene presa
                action[9] = 0.40   # Robot 1 solleva in Z
                action[13] = 1.0   # Mantiene presa

            ep_obs.append(flat_obs)
            ep_act.append(action)

            obs, reward, done, info = env.step(action)
            flat_obs = np.concatenate([
                obs["robot0_proprio-state"],
                obs["robot1_proprio-state"],
                obs["object-state"]
            ], dtype=np.float32)

        observations.extend(ep_obs)
        actions.extend(ep_act)
        pbar.update(1)

    pbar.close()
    env.close()

    os.makedirs("data", exist_ok=True)
    out_path = "data/robosuite_dual_dataset.npz"
    np.savez_compressed(
        out_path,
        obs=np.array(observations, dtype=np.float32),
        actions=np.array(actions, dtype=np.float32)
    )
    print(f"\n[+] Dataset generato: {len(observations)} transizioni salvate in {out_path}")

if __name__ == "__main__":
    main()
