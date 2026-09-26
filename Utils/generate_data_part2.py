"""
Generates the training data (boundary + initial condition) and collocation
points used by the parametric PINN in
PINNs_Part_2_Parametric_Problems.ipynb, for the diffusion equation

    du/dt = D (d^2u/dx^2 + d^2u/dy^2)

on the domain t in [0, 100], x, y in [-1, 1], D in [0.5, 1.5].

The exact solution used to label the boundary/initial points is

    u(t, x, y, D) = (x^2 - y^2) + exp(-2 pi^2 D t) sin(pi x) sin(pi y)

which is the steady state x^2 - y^2 from part 1 (itself an exact solution of
the diffusion equation, since it solves the Laplace equation) plus a
transient mode that decays away as t grows, at a rate controlled by D. On
the spatial boundary (x = +-1 or y = +-1) the transient term vanishes
exactly, so this reduces to the same x^2 - y^2 boundary values as part 1,
regardless of t or D.

* Training data (for L_data): points on the spatial boundary (x = -1, 1 or
  y = -1, 1) with random t and D, plus points at the temporal boundary
  (t = 0, the initial condition) with random x, y and D. Saved to
  training_data_part2.csv (columns: t, x, y, D, u).

* Collocation points (for L_pde): points scattered inside the open domain
  (0 < t < 100, -1 < x < 1, -1 < y < 1, 0.5 < D < 1.5) where u is unknown
  and the diffusion equation is enforced instead. Saved to
  collocation_points_part2.csv (columns: t, x, y, D) -- u is left out on
  purpose, since the PDE loss never uses a target value.

Run directly to (re)generate both CSV files:

    python generate_data_part2.py
"""

import os

import numpy as np
import pandas as pd
from scipy.stats import qmc

T_MIN, T_MAX = 0.0, 100.0
X_MIN, X_MAX = -1.0, 1.0
Y_MIN, Y_MAX = -1.0, 1.0
D_MIN, D_MAX = 0.5, 1.5


def exact_solution(t, x, y, D):
    return (x ** 2 - y ** 2) + np.exp(-2 * np.pi ** 2 * D * t) * np.sin(np.pi * x) * np.sin(np.pi * y)


def _scale(unit_cube, lows, highs):
    lows = np.asarray(lows)
    highs = np.asarray(highs)
    return unit_cube * (highs - lows) + lows


def generate_boundary_data(n_per_edge=150, n_ic=300, seed=0):
    rows = []

    # spatial boundary: x = -1, x = 1, y = -1, y = 1, each with random t, D
    # and a random position along the free coordinate
    engine = qmc.LatinHypercube(d=3, seed=seed)
    for edge_value, fixed_axis in [(-1.0, "x"), (1.0, "x"), (-1.0, "y"), (1.0, "y")]:
        sample = engine.random(n=n_per_edge)
        free, t, D = _scale(sample, [-1.0, T_MIN, D_MIN], [1.0, T_MAX, D_MAX]).T
        if fixed_axis == "x":
            x, y = np.full(n_per_edge, edge_value), free
        else:
            x, y = free, np.full(n_per_edge, edge_value)
        u = exact_solution(t, x, y, D)
        rows.append(pd.DataFrame({"t": t, "x": x, "y": y, "D": D, "u": u}))

    # temporal boundary: t = 0, any x, y, D -- the initial condition
    engine_ic = qmc.LatinHypercube(d=3, seed=seed + 1)
    sample = engine_ic.random(n=n_ic)
    x, y, D = _scale(sample, [X_MIN, Y_MIN, D_MIN], [X_MAX, Y_MAX, D_MAX]).T
    t = np.zeros(n_ic)
    u = exact_solution(t, x, y, D)
    rows.append(pd.DataFrame({"t": t, "x": x, "y": y, "D": D, "u": u}))

    return pd.concat(rows, ignore_index=True)


def generate_collocation_data(Nc=3000, seed=2):
    engine = qmc.LatinHypercube(d=4, seed=seed)
    sample = engine.random(n=Nc)
    t, x, y, D = _scale(sample, [T_MIN, X_MIN, Y_MIN, D_MIN], [T_MAX, X_MAX, Y_MAX, D_MAX]).T
    return pd.DataFrame({"t": t, "x": x, "y": y, "D": D})


if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))

    data_df = generate_boundary_data()
    data_df.to_csv(os.path.join(out_dir, "training_data_part2.csv"), index=False)

    colloc_df = generate_collocation_data()
    colloc_df.to_csv(os.path.join(out_dir, "collocation_points_part2.csv"), index=False)

    print(f"Saved {len(data_df)} boundary/initial points to training_data_part2.csv")
    print(f"Saved {len(colloc_df)} collocation points to collocation_points_part2.csv")
