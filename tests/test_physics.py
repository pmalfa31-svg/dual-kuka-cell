import os
import pytest
import mujoco
import numpy as np

def test_xml_load():
    xml_path = "assets/scene.xml"
    assert os.path.exists(xml_path), f"Scene file not found: {xml_path}"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    assert model.nq > 0
    assert model.nv > 0

def test_gravity_chute_dynamics():
    model = mujoco.MjModel.from_xml_path("assets/scene.xml")
    data = mujoco.MjData(model)
    
    # Check initial position on chute
    box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box_0")
    init_pos = data.xpos[box_id].copy()
    
    # Step simulation without robot inputs
    for _ in range(500):
        mujoco.mj_step(model, data)
        
    final_pos = data.xpos[box_id].copy()
    # Box must slide down along -X and decrease Z
    assert final_pos[0] < init_pos[0], "Package failed to slide down along X-axis"
    assert final_pos[2] < init_pos[2], "Package failed to decrease Z altitude"
