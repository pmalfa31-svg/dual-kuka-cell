import yaml
import numpy as np
import mujoco

with open("configs/training_cfg.yaml", "r") as f:
    cfg = yaml.safe_load(f)

model = mujoco.MjModel.from_xml_path(cfg["env"]["xml_path"])
data = mujoco.MjData(model)

site_r1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "r1_suction_site")
site_r2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "r2_suction_site")
box_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box_0")

pallet_1 = np.array(cfg["pallet_targets"]["robot_1"], dtype=np.float32)
pallet_2 = np.array(cfg["pallet_targets"]["robot_2"], dtype=np.float32)

print("\n" + "=" * 60)
print("          INSPECTION & WAYPOINT SOLVER MULTI-START          ")
print("=" * 60)

def solve_precise_ik(robot_id: int, target_xyz: np.ndarray, base_seed: np.ndarray = None) -> np.ndarray:
    q_idx = slice(0, 6) if robot_id == 1 else slice(6, 12)
    site_id = site_r1 if robot_id == 1 else site_r2
    low = model.jnt_range[q_idx, 0]
    high = model.jnt_range[q_idx, 1]

    seeds = []
    if base_seed is not None:
        seeds.append(base_seed.copy())
    seeds.append(np.array([0.0, -0.6, 1.6, 0.0, 0.6, 0.0]))
    seeds.append(np.array([-1.57, 0.3, 1.2, 0.0, 0.2, 0.0]))
    seeds.append(np.array([ 1.57, 0.3, 1.2, 0.0, 0.2, 0.0]))
    seeds.append(np.array([ 3.14, -0.5, 1.5, 0.0, 0.6, 0.0]))

    # Aggiunge seed casuali uniformi nello spazio dei giunti ammissibile
    np.random.seed(42)
    for _ in range(30):
        seeds.append(np.random.uniform(low, high))

    best_q = None
    best_err = float("inf")

    for seed in seeds:
        data.qpos[q_idx] = seed
        mujoco.mj_forward(model, data)

        for _ in range(800):
            current_xyz = data.site_xpos[site_id]
            err = target_xyz - current_xyz
            err_norm = np.linalg.norm(err)
            if err_norm < 1e-4:
                break

            jacp = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jacp, None, site_id)
            J = jacp[:, q_idx]

            # Damped Least Squares con raggio di convergenza adattivo
            damping = 1e-3 if err_norm < 0.05 else 1e-2
            dq = J.T @ np.linalg.inv(J @ J.T + damping * np.eye(3)) @ err
            step_size = min(0.08, err_norm * 0.8)
            dq = np.clip(dq, -step_size, step_size)

            data.qpos[q_idx] = np.clip(data.qpos[q_idx] + dq, low, high)
            mujoco.mj_forward(model, data)

        final_err = np.linalg.norm(target_xyz - data.site_xpos[site_id])
        if final_err < best_err:
            best_err = final_err
            best_q = data.qpos[q_idx].copy()
            if best_err < 1e-3:  # Sotto 1 mm
                break

    print(f" -> Robot {robot_id} to [{target_xyz[0]:.2f}, {target_xyz[1]:.2f}, {target_xyz[2]:.2f}] | Errore residuo: {best_err * 1000:.2f} mm")
    return best_q

# Posa di riposo nominale
q_home_1 = np.array([0.0, -0.6, 1.6, 0.0, 0.6, 0.0], dtype=np.float32)
q_home_2 = np.array([0.0, -0.6, 1.6, 0.0, 0.6, 0.0], dtype=np.float32)

print("\n--- Calcolo Pose Robot 1 (Nord) ---")
wp_home_1    = q_home_1.copy()
wp_prepick_1 = solve_precise_ik(1, np.array([0.0, 0.60, 0.62]), q_home_1)
wp_pick_1    = solve_precise_ik(1, np.array([0.0, 0.60, 0.50]), wp_prepick_1)
wp_lift_1    = wp_prepick_1.copy()
wp_place_1   = solve_precise_ik(1, np.array([pallet_1[0], pallet_1[1], 0.38]), wp_lift_1)

print("\n--- Calcolo Pose Robot 2 (Sud) ---")
wp_home_2    = q_home_2.copy()
wp_prepick_2 = solve_precise_ik(2, np.array([0.0, -0.60, 0.62]), q_home_2)
wp_pick_2    = solve_precise_ik(2, np.array([0.0, -0.60, 0.50]), wp_prepick_2)
wp_lift_2    = wp_prepick_2.copy()
wp_place_2   = solve_precise_ik(2, np.array([pallet_2[0], pallet_2[1], 0.38]), wp_lift_2)

np.savez(
    "data/calibrated_waypoints.npz",
    wp_r1=np.array([wp_home_1, wp_prepick_1, wp_pick_1, wp_lift_1, wp_place_1], dtype=np.float32),
    wp_r2=np.array([wp_home_2, wp_prepick_2, wp_pick_2, wp_lift_2, wp_place_2], dtype=np.float32)
)
print("\n[+] Waypoint salvati con successo in data/calibrated_waypoints.npz")
print("=" * 60)
