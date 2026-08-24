import mujoco
import numpy as np
from src.core.conveyor import OvalConveyor

def test_conveyor_vector_field():
    model = mujoco.MjModel.from_xml_path("assets/scene.xml")
    data = mujoco.MjData(model)
    conveyor = OvalConveyor(speed=0.6)
    
    # Manually place box on straight section (North)
    box_qpos_adr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "box_0_joint")]
    data.qpos[box_qpos_adr:box_qpos_adr+3] = [0.0, 0.6, 0.45]
    mujoco.mj_forward(model, data)
    
    conveyor.apply_conveyor_velocity(model, data, "box_0")
    
    jnt_qvel_adr = model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "box_0_joint")]
    vx = data.qvel[jnt_qvel_adr]
    assert vx < 0, f"Expected negative X velocity on North straight section, got {vx}"
