import yaml
import numpy as np
import mujoco

with open("configs/training_cfg.yaml", "r") as f:
    cfg = yaml.safe_load(f)

model = mujoco.MjModel.from_xml_path(cfg["env"]["xml_path"])
data = mujoco.MjData(model)

site_r1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "r1_suction_site")
site_r2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "r2_suction_site")

def solve_exact_joint_pose(robot_id: int, target_xyz: np.ndarray, q_seed: np.ndarray) -> np.ndarray:
    q_idx = slice(0, 6) if robot_id == 1 else slice(6, 12)
    site_id = site_r1 if robot_id == 1 else site_r2
    
    mujoco.mj_resetData(model, data)
    data.qpos[q_idx] = q_seed
    mujoco.mj_forward(model, data)

    for _ in range(1000):
        current_xyz = data.site_xpos[site_id]
        err = target_xyz - current_xyz
        if np.linalg.norm(err) < 1e-4:
            break
        
        jacp = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, None, site_id)
        J = jacp[:, q_idx]
        
        dq = J.T @ np.linalg.inv(J @ J.T + 1e-4 * np.eye(3)) @ err
        data.qpos[q_idx] += np.clip(dq, -0.05, 0.05)
        mujoco.mj_forward(model, data)

    return np.round(data.qpos[q_idx].copy(), 3)

q_seed = np.array([0.0, -0.6, 1.6, 0.0, 0.6, 0.0])

# Robot 1 (Nord)
q_standby_1 = solve_exact_joint_pose(1, np.array([0.0, 0.60, 0.65]), q_seed)
q_pick_1    = solve_exact_joint_pose(1, np.array([0.0, 0.60, 0.49]), q_seed)
q_pallet_1  = solve_exact_joint_pose(1, np.array([0.80, 1.00, 0.35]), q_seed)

# Robot 2 (Sud)
q_standby_2 = solve_exact_joint_pose(2, np.array([0.0, -0.60, 0.65]), q_seed)
q_pick_2    = solve_exact_joint_pose(2, np.array([0.0, -0.60, 0.49]), q_seed)
q_pallet_2  = solve_exact_joint_pose(2, np.array([-0.80, -1.00, 0.35]), q_seed)

print("\n" + "=" * 60)
print("           POSE GIUNTI CALIBRATE (RADIANTI)           ")
print("=" * 60)
print(f"R1_STANDBY = np.array({q_standby_1.tolist()})")
print(f"R1_PICK    = np.array({q_pick_1.tolist()})")
print(f"R1_PALLET  = np.array({q_pallet_1.tolist()})\n")
print(f"R2_STANDBY = np.array({q_standby_2.tolist()})")
print(f"R2_PICK    = np.array({q_pick_2.tolist()})")
print(f"R2_PALLET  = np.array({q_pallet_2.tolist()})")
print("=" * 60)
