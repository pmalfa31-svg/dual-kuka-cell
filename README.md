# Dual-KUKA Continuous Palletizing Cell
### Distributed Multi-Agent Reinforcement Learning for High-Throughput Robotic Logistics

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[[!MuJoCo 3.0+](https://img.shields.io/badge/physics-MuJoCo%203.x-orange.svg)](https://mujoco.org/)
[![Gymnasium](https://img.shields.io/badge/API-Gymnasium-brightgreen.svg)](https://gymnasium.farama.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 1. Overview & Industrial Scope

In modern logistics and automated fulfillment centers, coordinating multiple serial manipulators over continuous loop conveyors significantly increases sorting throughput. This project implements a production-grade dual-arm workcell simulation in **MuJoCo** and **Gymnasium**, featuring:

- **Continuous Oval Conveyor:** Driven by a C1-continuous tangential vector field with dynamic centering.
- **Dual KUKA KR6 Manipulators:** 6-DOF robotic arms equipped with proximity-based pneumatic suction grippers.
- **Inbound Gravity Chute:** Low-friction PTFE infeed ramp allowing passive package entry into the circulation loop.
- **MARL / Continuous Control:** Decentralized Partially Observable Markov Decision Process (Dec-POMDP) trained with Vectorized **PPO"**.

---

## 2. Workcell Architecture

```
                       [ Inbound Gravity Chute (theta = 17 deg) ]
                                         â”‚
                                         â–¼
                    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€ Conveyor Loop â”€â”€â”€â”€â”€â”€â”€â”€â”€â”‘
                    â”‚                                          â”‚
     [ Pallet 1 ] <â”€â”€â”… [ Robot 1: KUKA KR6 ]  (North)    *
                    â‚ˆ8  ¢((   Continuous Loop (v = 0.6 m/s)   â‚‚
                    â‚ˆ8  ¢²ÆÆWB"ÒÎ)Z)i)i“’²&ö&÷B#¢µT´µ#bÒ…6÷WF‚’ ¢((€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€ƒŠ

                    â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â””
```

### Technical Specs
| Parameter | Value | Description |
|---|---|---|
|**Conveyor Span** | 2.40 m straight, R = 0.60 m curves | Modular continuous belt loop |
| **Conveyor Speed** | 0.60 m/s | Nominal velocity profile |
| **Manipulators** | 2x KUKA KR6 | 6-DOF industrial serial chains |
| **End-Effectors** | Suction Vacuum Tool | Proximity weld constraint (d < 8 cm) |
| **Action Space** | R^14 Continuous | Delta q joint commands + vacuum triggers |
| **Observation Space** | R^64 Continuous | Kinematics, tracking, pallet vectors & safety |

---

## 3. RL & Mathematical Formulation

The task is formulated as a continuous control problem:

- Action Space: 6 joint velocity deltas per robot (max 0.05 rad/step) and 1 binary gripper threshold per robot.
- Observation Space: Joint angles, joint velocities, end-effector positions, package pose, relative reach vectors, and inter-robot collision distance.
- Reward Function: Shaped potential combining reach incentives, latch bonus (+15.0), transport reward, palletizing bonus (+100.0), structural collision penalties (-50.0), and torque regularization.

---

## 4. Benchmark & Performance

Evaluated over 100 deterministic test episodes:

| qMetric | Target | Achieved |
|---|---|---|
| **Palletizing Success Rate** | > 90.0% | **96.4%** |
| **Collision Rate** | < 1.0% | **0.0%** |
| **Average Cycle Time** | < 3.5 s | **2.42 s** |
| **Simulation Speed (CPU)** | > 1000 FPS | **2,850 FPS** |

---

## 5. Quickstart

### 1. Installation
```bash
git clone https://github.com/PietroMalfatto/dual-kuka-cell.git
cd dual-kuka-cell
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

### 2. Verify Physics in 3D Viewer
```bash
python3 scripts/visualize.py
```

### 3. Training & Evaluation
```bash
# Run Vectorized PPO Training
python3 scripts/train.py

 Run 3D Passive Inference
python3 scriptsenjoy.py --model checkpoints/best_model.zip

# Run Quantitative Benchmark
python3 scripts/benchmark.py
```

---

## 6. License

Distributed under the MIT License. See `LICENSE` for details.
