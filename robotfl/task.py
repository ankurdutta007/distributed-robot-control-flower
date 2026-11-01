"""
Shared code: robot-joint physics, expert controller, neural controller (pure NumPy),
data generation for each robot cell, and closed-loop evaluation.

Story: a robotic assembly line has N robot cells (stations). Each station handles a
DIFFERENT payload (part weight) and has its own joint friction. Raw sensor logs stay
inside each station (privacy / bandwidth). Stations only share neural-network weights,
and a Flower server aggregates them (FedAvg) into one global controller.
"""
import numpy as np

# ---------------- Physics of one robot joint (1-DOF arm carrying a payload) ----------------
G, L, J0, DT, UMAX = 9.81, 0.5, 0.05, 0.02, 20.0
KP, KD = 60.0, 10.0  # gains of the "expert" PD controller we learn from

# Each station: payload range (kg) and joint friction
STATIONS = [
    {"payload": (0.1, 0.5), "friction": 0.05},
    {"payload": (0.5, 0.9), "friction": 0.10},
    {"payload": (0.9, 1.3), "friction": 0.15},
    {"payload": (1.3, 1.7), "friction": 0.20},
    {"payload": (1.7, 2.1), "friction": 0.25},
    {"payload": (2.1, 2.5), "friction": 0.30},
]
EVAL_PAYLOADS = [0.3, 0.7, 1.1, 1.5, 1.9, 2.3]  # one per station range


def expert_action(theta_ref, theta, omega, m):
    """PD control + gravity compensation (what a classical controller would do)."""
    u = KP * (theta_ref - theta) - KD * omega + m * G * L * np.sin(theta)
    return np.clip(u, -UMAX, UMAX)


def step_dynamics(theta, omega, u, m, b):
    J = J0 + m * L**2
    alpha = (u - b * omega - m * G * L * np.sin(theta)) / J
    omega = omega + alpha * DT
    theta = theta + omega * DT
    return theta, omega


def features(theta_ref, theta, omega, m):
    """What the neural controller sees (all measurable by the robot)."""
    return np.stack([(theta_ref - theta) / 1.0, omega / 3.0, np.sin(theta), m / 2.0], axis=-1)


# ---------------- Neural controller (tiny MLP, NumPy only) ----------------
def init_params(seed=0, hidden=32):
    rng = np.random.default_rng(seed)
    return [
        rng.normal(0, 1 / np.sqrt(4), (4, hidden)), np.zeros(hidden),
        rng.normal(0, 1 / np.sqrt(hidden), (hidden, hidden)), np.zeros(hidden),
        rng.normal(0, 1 / np.sqrt(hidden), (hidden, 1)), np.zeros(1),
    ]


def forward(params, x):
    W1, b1, W2, b2, W3, b3 = params
    h1 = np.tanh(x @ W1 + b1)
    h2 = np.tanh(h1 @ W2 + b2)
    return (h2 @ W3 + b3), (x, h1, h2)


def policy_action(params, theta_ref, theta, omega, m):
    x = features(theta_ref, theta, omega, m)[None, :]
    y, _ = forward(params, x)
    return float(np.clip(y[0, 0] * UMAX, -UMAX, UMAX))


def train_epochs(params, X, Y, epochs, lr, batch=64, seed=0):
    """Mini-batch gradient descent with Adam-free simple SGD+momentum. Returns (params, last_loss)."""
    rng = np.random.default_rng(seed)
    params = [p.copy() for p in params]
    vel = [np.zeros_like(p) for p in params]
    n, loss = len(X), 0.0
    for _ in range(epochs):
        idx = rng.permutation(n)
        for s in range(0, n, batch):
            bi = idx[s:s + batch]
            x, y = X[bi], Y[bi]
            pred, (xin, h1, h2) = forward(params, x)
            err = pred - y
            loss = float(np.mean(err**2))
            d3 = 2 * err / len(bi)
            gW3, gb3 = h2.T @ d3, d3.sum(0)
            d2 = (d3 @ params[4].T) * (1 - h2**2)
            gW2, gb2 = h1.T @ d2, d2.sum(0)
            d1 = (d2 @ params[2].T) * (1 - h1**2)
            gW1, gb1 = xin.T @ d1, d1.sum(0)
            grads = [gW1, gb1, gW2, gb2, gW3, gb3]
            for i in range(6):
                vel[i] = 0.9 * vel[i] - lr * grads[i]
                params[i] += vel[i]
    return params, loss


def mse(params, X, Y):
    pred, _ = forward(params, X)
    return float(np.mean((pred - Y) ** 2))


# ---------------- Local data of one robot cell ----------------
def make_station_data(station_id, n=4000, seed=0):
    """Simulated sensor log from station `station_id`: states visited while the expert
    controller works (with exploration noise) + expert's action as the label."""
    st = STATIONS[station_id]
    rng = np.random.default_rng(1000 * seed + station_id)
    m = rng.uniform(*st["payload"], n)
    ref = rng.uniform(-1.2, 1.2, n)
    theta = ref + rng.uniform(-1.5, 1.5, n)
    omega = rng.uniform(-3, 3, n)
    u = expert_action(ref, theta, omega, m)
    X = features(ref, theta, omega, m)
    Y = (u / UMAX)[:, None]
    return X, Y


# ---------------- Closed-loop test: does the learned controller really move the arm? ----------------
def rollout(controller, m, b=0.15, ref=1.0, T=3.0, theta0=0.0):
    steps = int(T / DT)
    th, om = theta0, 0.0
    ths = []
    for _ in range(steps):
        u = controller(ref, th, om, m)
        th, om = step_dynamics(th, om, u, m, b)
        ths.append(th)
    ths = np.array(ths)
    return ths


def tracking_rmse(ths, ref=1.0):
    return float(np.sqrt(np.mean((ths - ref) ** 2)))


def evaluate_controller(controller, payloads=EVAL_PAYLOADS):
    """Average step-response tracking RMSE over payloads (and two targets)."""
    out = {}
    for m in payloads:
        b = 0.05 + 0.1 * m
        r = [tracking_rmse(rollout(controller, m, b, ref=ref), ref) for ref in (1.0, -0.8)]
        out[m] = float(np.mean(r))
    return out
