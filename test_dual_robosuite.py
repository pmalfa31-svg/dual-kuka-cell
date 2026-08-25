import robosuite as suite
import numpy as np

def main():
    print("[*] Inizializzazione cella Dual-Arm KUKA IIWA...")
    
    # Creiamo l'ambiente bimanuale con 2 KUKA IIWA
    env = suite.make(
        env_name="TwoArmLift",
        robots=["IIWA", "IIWA"],
        has_renderer=True,
        has_offscreen_renderer=False,
        control_freq=20,
        horizon=200,
        use_camera_obs=False
    )

    obs = env.reset()
    print("[+] Cella bimanuale caricata con successo!")
    print(f"[+] Dimensione spazio d'azione complessivo: {env.action_dim}D (7D Robot 1 + 7D Robot 2)")

    # Eseguiamo una sequenza di controllo coordinato
    for step in range(200):
        # Azioni in spazio operativo [-1, 1] per ciascun braccio (Delta-XYZ, Rotazioni, Gripper)
        action = np.zeros(env.action_dim, dtype=np.float32)
        
        # Movimento sinusoidale dolce sui due end-effector
        t = step * 0.1
        action[0] = 0.3 * np.sin(t)        # Delta X Robot 1
        action[2] = 0.2 * np.cos(t)        # Delta Z Robot 1
        action[6] = -1.0                   # Gripper 1 Aperto
        
        action[7] = -0.3 * np.sin(t)       # Delta X Robot 2
        action[9] = 0.2 * np.cos(t)        # Delta Z Robot 2
        action[13] = -1.0                  # Gripper 2 Aperto

        obs, reward, done, info = env.step(action)
        env.render()

    env.close()
    print("[+] Test Dual-Arm completato con successo!")

if __name__ == "__main__":
    main()
