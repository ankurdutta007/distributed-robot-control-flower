"""Run AFTER `flwr run`. Compares: expert PD, Federated (Flower), Local-only (each robot alone),
and makes the plots used in the report."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from robotfl.task import (EVAL_PAYLOADS, STATIONS, evaluate_controller, expert_action, init_params,
                          make_station_data, policy_action, rollout, train_epochs)

os.makedirs("results", exist_ok=True)
ROUNDS, EPOCHS, LR = 30, 5, 0.01

# 1) Federated model (saved by the Flower server)
npz = np.load("results/global_model.npz")
fed = [npz[f"arr_{i}"] for i in range(len(npz.files))]
fed_ctrl = lambda r, t, o, m: policy_action(fed, r, t, o, m)
expert_ctrl = lambda r, t, o, m: float(expert_action(r, t, o, m))

# 2) Local-only baselines: each station trains alone with the SAME total compute
local_models = []
for s in range(len(STATIONS)):
    X, Y = make_station_data(s)
    p, _ = train_epochs(init_params(0), X, Y, epochs=ROUNDS * EPOCHS, lr=LR, seed=s)
    local_models.append(p)

# 3) Centralized (everyone uploads raw data) - upper bound, but violates data privacy
Xc = np.concatenate([make_station_data(s)[0] for s in range(6)])
Yc = np.concatenate([make_station_data(s)[1] for s in range(6)])
cent, _ = train_epochs(init_params(0), Xc, Yc, epochs=ROUNDS * EPOCHS // 6 + 5, lr=LR, seed=1)

def eval_params(p):
    return evaluate_controller(lambda r, t, o, m: policy_action(p, r, t, o, m))

res = {"Expert PD (reference)": evaluate_controller(expert_ctrl),
       "Federated (Flower)": evaluate_controller(fed_ctrl),
       "Centralized (raw data shared)": eval_params(cent)}
for s, p in enumerate(local_models):
    res[f"Local-only: Station {s+1}"] = eval_params(p)

table = {k: {"per_payload": {str(m): v for m, v in d.items()}, "avg": float(np.mean(list(d.values())))}
         for k, d in res.items()}
json.dump(table, open("results/comparison.json", "w"), indent=2)
print(f"{'Method':35s}{'Avg RMSE':>10s}")
for k, v in table.items():
    print(f"{k:35s}{v['avg']:10.3f}")

# ---- Plot 1: learning curve
h = json.load(open("results/fed_history.json"))
plt.figure(figsize=(6, 4))
plt.plot([x["round"] for x in h[1:]], [x["avg_rmse"] for x in h[1:]], "o-")
plt.axhline(table["Expert PD (reference)"]["avg"], color="g", ls="--", label="Expert PD")
plt.xlabel("Federated round"); plt.ylabel("Avg tracking RMSE (rad)")
plt.title("Global controller improves each round"); plt.legend(); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig("results/fig1_learning_curve.png", dpi=150); plt.close()

# ---- Plot 2: bar chart, per payload
plt.figure(figsize=(8, 4.5))
locals_avg = np.mean([[res[f"Local-only: Station {s+1}"][m] for m in EVAL_PAYLOADS] for s in range(6)], axis=0)
w = 0.27
x = np.arange(len(EVAL_PAYLOADS))
plt.bar(x - w, [res["Expert PD (reference)"][m] for m in EVAL_PAYLOADS], w, label="Expert PD")
plt.bar(x, [res["Federated (Flower)"][m] for m in EVAL_PAYLOADS], w, label="Federated")
plt.bar(x + w, locals_avg, w, label="Local-only (avg of 6 robots)")
plt.xticks(x, [f"{m} kg" for m in EVAL_PAYLOADS]); plt.ylabel("Tracking RMSE (rad)")
plt.xlabel("Payload"); plt.title("Federated controller works for ALL payloads; local-only does not")
plt.yscale("log"); plt.legend(); plt.tight_layout(); plt.savefig("results/fig2_payload_comparison.png", dpi=150); plt.close()

# ---- Plot 3: step responses, heavy payload (2.3 kg)
t = np.arange(150) * 0.02
plt.figure(figsize=(6.5, 4))
m = 2.3
for name, c in [("Expert PD", expert_ctrl), ("Federated", fed_ctrl),
                ("Local-only (Station 1, light payloads)", lambda r, th, o, mm: policy_action(local_models[0], r, th, o, mm))]:
    plt.plot(t, rollout(c, m, 0.05 + 0.1 * m), label=name)
plt.axhline(1.0, color="k", ls=":", label="Target")
plt.ylim(-1.5, 2.5); plt.xlabel("Time (s)"); plt.ylabel("Joint angle (rad)")
plt.title("Step response with 2.3 kg payload"); plt.legend(fontsize=8); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig("results/fig3_step_response.png", dpi=150); plt.close()
print("Saved figures in results/")
