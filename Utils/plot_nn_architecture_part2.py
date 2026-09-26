"""
Draws the fully-connected architecture used by the parametric PINN in
PINNs_Part_2_Parametric_Problems.ipynb:

    4 input neurons (t, x, y, D) -> 5 hidden layers of 40 tanh neurons each
    -> 1 output neuron (u)

Compared to the network in part 1, the input layer grew from 2 to 4 neurons
(t and D joined x and y) and the hidden layers were widened from 20 to 40
neurons, since the network now has to learn how the solution changes with
both time and diffusivity. The figure is saved as nn_architecture_part2.png
in this folder.

Run directly to (re)generate the image:

    python plot_nn_architecture_part2.py
"""

import os

import matplotlib.pyplot as plt

# layer sizes: input, 5 hidden layers, output
LAYER_SIZES = [4, 40, 40, 40, 40, 40, 1]
LAYER_LABELS = ["Input\n(t, x, y, D)"] + ["Hidden\n(tanh)"] * 5 + ["Output\n(u)"]


def plot_architecture(layer_sizes=LAYER_SIZES, layer_labels=LAYER_LABELS,
                       max_drawn_neurons=10, neuron_radius=0.3,
                       layer_spacing=2.5, neuron_spacing=0.8):
    """Draw a fully-connected feed-forward network diagram with pyplot.

    Layers wider than ``max_drawn_neurons`` are drawn with a fixed number of
    neurons plus a "..." marker, so a 40-neuron hidden layer stays readable.
    """
    n_layers = len(layer_sizes)

    style = open("plot_style.py").read()
    exec(style)

    fig, ax = plt.subplots(figsize=(2.2 * n_layers, 6))

    # for each layer, figure out how many neurons to actually draw and at
    # what y-positions, so wide layers are truncated with a "..." gap
    drawn_positions = []
    for size in layer_sizes:
        if size <= max_drawn_neurons:
            n_drawn = size
            ys = [(i - (n_drawn - 1) / 2) * neuron_spacing for i in range(n_drawn)]
            drawn_positions.append((ys, None))
        else:
            n_drawn = max_drawn_neurons
            ys = [(i - (n_drawn - 1) / 2) * neuron_spacing for i in range(n_drawn)]
            gap_index = n_drawn // 2
            ys[gap_index] += neuron_spacing * 0.6  # open a gap for the "..." label
            for i in range(gap_index + 1, n_drawn):
                ys[i] += neuron_spacing * 0.6
            drawn_positions.append((ys, gap_index))

    xs = [i * layer_spacing for i in range(n_layers)]

    # connections between consecutive layers
    for l in range(n_layers - 1):
        ys0, _ = drawn_positions[l]
        ys1, _ = drawn_positions[l + 1]
        for y0 in ys0:
            for y1 in ys1:
                ax.plot([xs[l], xs[l + 1]], [y0, y1],
                        color="gray", linewidth=0.4, alpha=0.5, zorder=1)

    # neurons
    for l, (ys, gap_index) in enumerate(drawn_positions):
        for i, y in enumerate(ys):
            circle = plt.Circle((xs[l], y), neuron_radius,
                                 color="#4C72B0" if 0 < l < n_layers - 1 else
                                 ("#55A868" if l == 0 else "#C44E52"),
                                 ec="black", zorder=3)
            ax.add_patch(circle)
            if gap_index is not None and i == gap_index:
                ax.text(xs[l], y - neuron_spacing * 0.35, "...",
                        ha="center", va="center", fontsize=14, zorder=4)

        # layer label (below the layer) and neuron count (above)
        ax.text(xs[l], min(ys) - neuron_spacing * 1.3, layer_labels[l],
                ha="center", va="top", fontsize=11)
        ax.text(xs[l], max(ys) + neuron_spacing * 1.1,
                f"{layer_sizes[l]} neuron" + ("s" if layer_sizes[l] != 1 else ""),
                ha="center", va="bottom", fontsize=9, color="dimgray")

    ax.set_xlim(min(xs) - 1.5, max(xs) + 1.5)
    all_ys = [y for ys, _ in drawn_positions for y in ys]
    ax.set_ylim(min(all_ys) - 2.2, max(all_ys) + 2.0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("PINN architecture: (t, x, y, D) $\\rightarrow$ u(t, x, y, D)", fontsize=14)

    fig.tight_layout()
    return fig, ax


if __name__ == "__main__":
    fig, ax = plot_architecture()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nn_architecture_part2.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved {out_path}")
