"""Flower server = the central coordinator. Aggregates weights with FedAvg and logs progress."""
import json
import os

import numpy as np
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from robotfl.task import init_params, policy_action, evaluate_controller

app = ServerApp()
HISTORY = []


def global_eval(server_round: int, arrays: ArrayRecord):
    """After every round: test the global controller in closed loop on all payloads."""
    params = arrays.to_numpy_ndarrays()
    ctrl = lambda ref, th, om, m: policy_action(params, ref, th, om, m)
    res = evaluate_controller(ctrl)
    avg = float(np.mean(list(res.values())))
    HISTORY.append({"round": server_round, "avg_rmse": avg, "per_payload": {str(k): v for k, v in res.items()}})
    os.makedirs("results", exist_ok=True)
    json.dump(HISTORY, open("results/fed_history.json", "w"), indent=2)
    np.savez("results/global_model.npz", *params)
    print(f"[Round {server_round}] global controller tracking RMSE = {avg:.4f}")
    return MetricRecord({"avg_rmse": avg})


@app.main()
def main(grid: Grid, context: Context) -> None:
    rounds = int(context.run_config["num-server-rounds"])
    lr = float(context.run_config["lr"])
    HISTORY.clear()

    initial = ArrayRecord(init_params(seed=0))
    strategy = FedAvg(fraction_train=1.0, fraction_evaluate=1.0,
                      min_train_nodes=6, min_evaluate_nodes=6, min_available_nodes=6)
    strategy.start(
        grid=grid,
        initial_arrays=initial,
        num_rounds=rounds,
        train_config=ConfigRecord({"lr": lr}),
        evaluate_fn=global_eval,
    )
