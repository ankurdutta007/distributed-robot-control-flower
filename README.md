# Distributed (Federated) Control System for a Robotic Manufacturing Assembly

**Course:** Distributed Analytics for Industrial Automation (IM60210), Autumn 2026-27
**Student:** Ankur Dutta (22IM30044) | **Technology:** Flower (flwr) | **Language:** Python (NumPy)

## 1. Problem
A robotic assembly line has several robot cells. Each cell handles a different part (payload 0.1 to 2.5 kg)
and has different joint friction. A classical controller must be hand-tuned per cell. Instead, we let each
robot **learn a neural controller from its own sensor logs**, and make the cells **collaborate without sharing raw
data** (factory data is private and bandwidth-heavy). Only model weights travel to a central server.

## 2. Approach
| Component | What it is |
|---|---|
| Plant | 1-DOF robot joint with payload, gravity and friction (`robotfl/task.py`) |
| Expert | PD controller with gravity compensation; its actions are the training labels |
| Learner | Small neural network (4 inputs, 2 hidden layers of 32, 1 output torque), pure NumPy |
| Distributed layer | **Flower** ServerApp (FedAvg aggregation) + 6 ClientApps = 6 robot cells |
| Non-IID data | Each station only sees its own payload range, so no single cell sees everything |

Algorithm: **Federated Averaging (FedAvg)**, McMahan et al., 2017. Framework: Flower, Beutel et al., 2020.

Per round: server sends global weights, each robot trains 5 local epochs on private data, the server
averages the weights, then we test the global controller in a **closed-loop simulation** (step response).

## 3. Results (30 rounds, 6 robot cells)
Metric: tracking RMSE (rad) of a step response, averaged over 6 payloads (0.3 to 2.3 kg). Lower is better.

| Method | Avg RMSE |
|---|---|
| Expert PD (reference) | 0.165 |
| **Federated (Flower), no raw data shared** | **0.168** |
| Centralized (all raw data pooled, not private) | 0.166 |
| Local-only, Station 1 (light payloads) | 0.187 |
| Local-only, Station 2 | 0.175 |
| Local-only, Station 3 | 0.171 |
| Local-only, Station 4 | 0.169 |
| Local-only, Station 5 | 0.173 |
| Local-only, Station 6 | 0.178 |

Findings: the federated controller is within about 2% of the expert and of the centralized upper bound, without any
raw data leaving a station. Every robot trained alone is worse on average, and the worst case is
a robot tested on payloads outside its own range (see `fig2`, `fig3`).

Figures: `results/fig1_learning_curve.png`, `fig2_payload_comparison.png`, `fig3_step_response.png`.

## 4. How to run
```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -e .                                      # installs flwr, numpy, matplotlib
flwr run . --federation-config "num-supernodes=6"     # runs the federated training (about 1-2 min)
python analyze.py                                     # baselines + plots -> results/
```
Settings (rounds, local epochs, learning rate) are in `pyproject.toml`. If Flower complains about CPUs,
add `client-resources-num-cpus=1` inside the quotes after `num-supernodes=6`.

## 5. Files
```
robotfl/task.py        physics, expert, neural net, data, closed-loop evaluation
robotfl/client_app.py  one robot cell (local training)
robotfl/server_app.py  coordinator (FedAvg + per-round evaluation)
analyze.py             baselines and figures
results/               saved metrics and plots
```

## 6. Limitations and future work
Simulated 1-DOF plant (not real hardware); no communication delays or dropped clients; could add differential
privacy, FedProx for stronger heterogeneity, and a multi-joint arm.

## References
1. McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data", AISTATS 2017.
2. Beutel et al., "Flower: A Friendly Federated Learning Framework", arXiv:2007.14390, 2020.
