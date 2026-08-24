import os
import mujoco
import numpy as np

def test_xml_load():
    xml_path = "assets/scene.xml"
    assert os.path.exists(xml_path), f"File non trovato: {xml_path}"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    assert model.nq > 0
    assert model.nv > 0

def test_chute_gravity():
    model = mujoco.MjModel.from_xml_path("assets/scene.xml")
    data = mujoco.MjData(model)
    
    # Inizializza la cinematica diretta
    mujoco.mj_forward(model, data)
    
    box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box_0")
    init_x = float(data.xpos[box_id][0])
    
    # Esegui 300 passi di dinamica
    for _ in range(300):
        mujoco.mj_step(model, data)
        
    final_x = float(data.xpos[box_id][0])
    # Il pacco deve scendere lungo -X rispetto alla posa iniziale
    assert final_x < init_x, f"Il pacco non scivola: init_x={init_x}, final_x={final_x}"
