"""
Generates the training data (boundary + initial condition), sparse sensor
data, and collocation points used by the inverse-problem PINN in
PINNs_Part_3_Inverse_Problems.ipynb, for the diffusion equation

    du/dt = D (d^2u/dx^2 + d^2u/dy^2)

on the domain t in [0, 100], x, y in [-1, 1], with a single, fixed but
*unknown-to-the-network* diffusivity D_TRUE.

The exact solution used to label every point is

    u(t, x, y, D) = (x^2 - y^2) + exp(-2 pi^2 D t) sin(pi x) sin(pi y)

same as in part 2, now evaluated at the one true value of D instead of a
whole range.

* Training data (for L_data): points on the spatial boundary (x = -1, 1 or
  y = -1, 1) with random t, plus points at the temporal boundary (t = 0, the
  initial condition) with random x, y. Saved to training_data_part3.csv
  (columns: t, x, y, u). Notice this data carries *no* information about D:
  on the spatial boundary sin(pi x) sin(pi y) is exactly zero, and at t = 0
  the transient factor exp(-2 pi^2 D t) is exactly 1, regardless of D.

* Sensor data (for L_sensor): a handful of interior "thermometer" readings,
  taken early (small t, before the transient has fully decayed away) so
  that they actually carry information about D. Saved to
  sensor_data_part3.csv (columns: t, x, y, u).

* Collocation points (for L_pde): points scattered inside the open domain
  (0 < t < 100, -1 < x < 1, -1 < y < 1) where u is unknown and the diffusion
  equation is enforced instead, using the network's current estimate of D.
  Saved to collocation_points_part3.csv (columns: t, x, y) -- u is left out
  on purpose, since the PDE loss never uses a target value.

Run directly to (re)generate all three CSV files:

    python generate_data_part3.py
"""

import os

import numpy as np
import pandas as pd
from scipy.stats import qmc

T_MIN, T_MAX = 0.0, 100.0
X_MIN, X_MAX = -1.0, 1.0
Y_MIN, Y_MAX = -1.0, 1.0

# the diffusivity of this particular piece of metal -- known to whoever
# generated this data, but hidden from the network, which only ever sees
# the columns saved to the CSV files below
D_TRUE = 0.8

# sensors only carry information about D while the transient term hasn't
# decayed away yet, so they are taken early: 2 pi^2 D_TRUE is already
# around 16, so by t ~ 0.3 the transient has all but vanished
T_SENSOR_MAX = 0.3


def exact_solution(t, x, y, D=D_TRUE):
    return (x ** 2 - y ** 2) + np.exp(-2 * np.pi ** 2 * D * t) * np.sin(np.pi * x) * np.sin(np.pi * y)


def _scale(unit_cube, lows, highs):
    lows = np.asarray(lows)
    highs = np.asarray(highs)
    return unit_cube * (highs - lows) + lows


def generate_boundary_data(n_per_edge=150, n_ic=300, seed=0):
    rows = []

    # spatial boundary: x = -1, x = 1, y = -1, y = 1, each with random t
    # and a random position along the free coordinate
    engine = qmc.LatinHypercube(d=2, seed=seed)
    for edge_value, fixed_axis in [(-1.0, "x"), (1.0, "x"), (-1.0, "y"), (1.0, "y")]:
        sample = engine.random(n=n_per_edge)
        free, t = _scale(sample, [-1.0, T_MIN], [1.0, T_MAX]).T
        if fixed_axis == "x":
            x, y = np.full(n_per_edge, edge_value), free
        else:
            x, y = free, np.full(n_per_edge, edge_value)
        u = exact_solution(t, x, y)
        rows.append(pd.DataFrame({"t": t, "x": x, "y": y, "u": u}))

    # temporal boundary: t = 0, any x, y -- the initial condition
    engine_ic = qmc.LatinHypercube(d=2, seed=seed + 1)
    sample = engine_ic.random(n=n_ic)
    x, y = _scale(sample, [X_MIN, Y_MIN], [X_MAX, Y_MAX]).T
    t = np.zeros(n_ic)
    u = exact_solution(t, x, y)
    rows.append(pd.DataFrame({"t": t, "x": x, "y": y, "u": u}))

    return pd.concat(rows, ignore_index=True)


def generate_sensor_data(n_sensors=40, seed=7):
    # a few thermometers pushed into the interior of the metal, read out
    # early in the diffusion process while the transient (and hence the
    # signature of D) is still visible
    engine = qmc.LatinHypercube(d=3, seed=seed)
    sample = engine.random(n=n_sensors)
    t, x, y = _scale(sample, [0.0, X_MIN, Y_MIN], [T_SENSOR_MAX, X_MAX, Y_MAX]).T
    u = exact_solution(t, x, y)
    return pd.DataFrame({"t": t, "x": x, "y": y, "u": u})


def generate_collocation_data(Nc=3000, seed=2, t_bias_power=4.0):
    # away from the transient (t >> 0.3 or so) the equation is satisfied by
    # u = x^2 - y^2 for *any* D, since both u_t and the Laplacian vanish
    # there -- so collocation points spread uniformly over t in [0, 100]
    # would spend >99% of their gradient signal on points that tell the
    # optimizer nothing about D. Skewing the t sample towards small t (via
    # a power-law transform of the uniform LHS sample) keeps points spread
    # across the whole domain while concentrating enough of them where the
    # PDE residual is actually sensitive to D.
    engine = qmc.LatinHypercube(d=3, seed=seed)
    sample = engine.random(n=Nc)
    sample[:, 0] = sample[:, 0] ** t_bias_power
    t, x, y = _scale(sample, [T_MIN, X_MIN, Y_MIN], [T_MAX, X_MAX, Y_MAX]).T
    return pd.DataFrame({"t": t, "x": x, "y": y})


if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))

    data_df = generate_boundary_data()
    data_df.to_csv(os.path.join(out_dir, "training_data_part3.csv"), index=False)

    sensor_df = generate_sensor_data()
    sensor_df.to_csv(os.path.join(out_dir, "sensor_data_part3.csv"), index=False)

    colloc_df = generate_collocation_data()
    colloc_df.to_csv(os.path.join(out_dir, "collocation_points_part3.csv"), index=False)

    print(f"True D used to generate this data: {D_TRUE}")
    print(f"Saved {len(data_df)} boundary/initial points to training_data_part3.csv")
    print(f"Saved {len(sensor_df)} sensor points to sensor_data_part3.csv")
    print(f"Saved {len(colloc_df)} collocation points to collocation_points_part3.csv")
