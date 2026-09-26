"""
Generates the two collocations of training points used by the PINN in
PINNs.ipynb and saves them to CSV files in this folder:

* Boundary data points (for the data loss L_data): 30 points on each of the
  4 edges of [-1, 1]^2, with the known exact solution u(x, y) = x^2 - y^2
  evaluated on the boundary. Saved to training_data.csv (columns: x, y, u).

* Collocation points (for the PDE loss L_pde): 400 points scattered inside
  the domain, where u is unknown and the Laplace equation is enforced
  instead. Saved to collocation_points.csv (columns: x, y) — u is left out
  on purpose, since the PDE loss never uses a target value.

The two are also merged into regular_neural_net_data.csv (columns: x, y, u),
with u filled in for every row (using the exact solution at the collocation
points too), since a regular, non-physics-informed network needs a label
everywhere.

For each CSV file, a matching PNG is also saved showing the (x, y) markers.

Run directly to (re)generate all three CSV and PNG files:

    python generate_data.py
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import qmc


def exact_solution(x, y):
    return x ** 2 - y ** 2


def generate_boundary_data(n_bc=4, n_data_per_bc=30):
    engine = qmc.LatinHypercube(d=1)
    data = np.zeros([n_bc, n_data_per_bc, 3])

    for i, j in zip(range(n_bc), [-1, +1, -1, +1]):
        points = (engine.random(n=n_data_per_bc)[:, 0] - 0.5) * 2
        if i < 2:
            data[i, :, 0] = j
            data[i, :, 1] = points
        else:
            data[i, :, 0] = points
            data[i, :, 1] = j

    # exact solution u(x, y) = x^2 - y^2 evaluated on the boundary
    for j in range(n_data_per_bc):
        data[0, j, 2] = -data[0, j, 1] ** 2 + 1
        data[1, j, 2] = -data[1, j, 1] ** 2 + 1
    for i in range(n_data_per_bc):
        data[2, i, 2] = data[2, i, 0] ** 2 - 1
        data[3, i, 2] = data[3, i, 0] ** 2 - 1

    data = data.reshape(n_data_per_bc * n_bc, 3)
    return pd.DataFrame(data, columns=["x", "y", "u"])


def generate_collocation_data(Nc=400):
    engine = qmc.LatinHypercube(d=2)
    colloc = engine.random(n=Nc)
    colloc = 2 * (colloc - 0.5)
    return pd.DataFrame(colloc, columns=["x", "y"])


def plot_points(df, title, png_path, marker_color="r"):
    style = open("plot_style.py").read()
    exec(style)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(df["x"], df["y"], s=10, marker="x", c=marker_color)
    ax.set_xlabel("x", fontsize=14)
    ax.set_ylabel("y", fontsize=14)
    ax.set_title(title, fontsize=14)
    ax.axis("square")
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))

    data_df = generate_boundary_data()
    data_df.to_csv(os.path.join(out_dir, "training_data.csv"), index=False)
    plot_points(data_df, "Training data (boundary points)",
                os.path.join(out_dir, "training_data.png"))

    pde_df = generate_collocation_data()
    pde_df.to_csv(os.path.join(out_dir, "collocation_points.csv"), index=False)
    plot_points(pde_df, "Collocation points",
                os.path.join(out_dir, "collocation_points.png"), marker_color="r")

    # a regular (non-physics) net needs a real label everywhere, so fill in
    # the exact solution at the collocation points too
    labeled_pde_df = pde_df.assign(u=exact_solution(pde_df["x"], pde_df["y"]))
    regular_df = pd.concat([data_df, labeled_pde_df], ignore_index=True)
    regular_df.to_csv(os.path.join(out_dir, "regular_neural_net_data.csv"), index=False)
    plot_points(regular_df, "Regular neural net data (all points)",
                os.path.join(out_dir, "regular_neural_net_data.png"), marker_color="r")

    print(f"Saved {len(data_df)} boundary points to training_data.csv/.png")
    print(f"Saved {len(pde_df)} collocation points to collocation_points.csv/.png")
    print(f"Saved {len(regular_df)} fully-labeled points to regular_neural_net_data.csv/.png")
