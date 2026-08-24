import numpy as np
import mujoco
from src.envs.dual_arm_env import DualKukaPalletizeEnv
from src.core.expert import IndustrialExpertPlanner

def main():
    env = DualKukaPalletizeEnv()
    expert = IndustrialExpertPlanner(env.pallet_1, env.pallet_2)

    obs, _ = env.reset()
    expert.reset()
    expert.active_robot = 1

    box_jnt_adr = env.model.jnt_qposadr[mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "box_0_joint")]
    env.data.qpos[box_jnt_adr:box_jnt_adr+3] = [0.0, 0.60, 0.46]
    env.data.qpos[box_jnt_adr+3:box_jnt_adr+7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(env.model, env.data)

    print("=" * 80)
    print("                VERIFICA CHIUSURA CICLO COMPLETO                ")
    print("=" * 80)

    for step_i in range(1, 150):
        action = expert.get_expert_action(env)
        obs, reward, terminated, truncated, info = env.step(action)

        if step_i % 10 == 0 or info["is_success"]:
            p_box, _ = env.kinematics.get_box_pose()
            d_pallet = np.linalg.norm(p_box[0:2] - env.pallet_1[0:2])
            print(f"Frame {step_i:3d} | State: {expert.state_r1} | Hold1: {str(info['holding_r1']):<5} | Dist Pallet: {d_pallet:.3f} m | Box Z: {p_box[2]:.2f} m")

        if info["is_success"]:
            print("=" * 80)
            print(f"[+] SUCCESSO CONFERMATO al frame {step_i}!")
            print("=" * 80)
            return

    print("[-] Non completato entro 150 frame.")

if __name__ == "__main__":
    main()
