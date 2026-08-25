import robosuite as suite
import numpy as np

def main():
    print("[*] Avvio Robosuite sul main...")
    
    # Rimuoviamo il controller_config esplicito e usiamo il default nativo di Robosuite 1.5
    env = suite.make(
        env_name="PickPlace",
        robots="IIWA",
        has_renderer=True,
        has_offscreen_renderer=False,
        control_freq=20,
        horizon=150,
        use_camera_obs=False
    )

    obs = env.reset()
    print("[+] Ambiente operativo! Finestra di simulazione attiva.")
    print(f"[+] Dimensione spazio d'azione nativo: {env.action_dim}D")

    for _ in range(150):
        # Generiamo azioni casuali per far muovere i giunti
        action = np.random.uniform(-1.0, 1.0, env.action_dim)
        env.step(action)
        env.render()

    env.close()
    print("[+] Test visivo completato con successo su main.")

if __name__ == "__main__":
    main()
