"""
Draws the fully-connected architecture used by the inverse-problem PINN in
PINNs_Part_3_Inverse_Problems.ipynb:

    3 input neurons (t, x, y) -> 5 hidden layers of 20 tanh neurons each
    -> 1 output neuron (u)

plus a separate learnable scalar D, drawn off to the side, since D is *not*
a network input here (unlike part 2's parametric PINN). D never touches the
input layer; it only ever appears inside the physics residual, multiplying
the Laplacian of u. The figure is saved as nn_architecture_part3.png in
this folder.

Run directly to (re)generate the image:

    python plot_nn_architecture_part3.py
"""

import os

import matplotlib.pyplot as plt

# layer sizes: input, 5 hidden layers, output
LAYER_SIZES = [3, 20, 20, 20, 20, 20, 1]
LAYER_LABELS = ["Input\n(t, x, y)"] + ["Hidden\n(tanh)"] * 5 + ["Output\n(u)"]


def plot_architecture(layer_sizes=LAYER_SIZES, layer_labels=LAYER_LABELS,
                       max_drawn_neurons=10, neuron_radius=0.3,
                       layer_spacing=2.5, neuron_spacing=0.8):
    """Draw a fully-connected feed-forward network diagram with pyplot.

    Layers wider than ``max_drawn_neurons`` are drawn with a fixed number of
    neurons plus a "..." marker, so a 20-neuron hidden layer stays readable.
    A separate, unconnected node for the learnable scalar D is drawn below
    the network to make clear it is not part of the input layer.
    """
    n_layers = len(layer_sizes)

    style = open("plot_style.py").read()
    exec(style)

    fig, ax = plt.subplots(figsize=(2.2 * n_layers, 6.5))

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

    all_ys = [y for ys, _ in drawn_positions for y in ys]

    # the learnable scalar D: drawn well below the network, with a dashed
    # line to the output neuron to show it joins u only inside the physics
    # residual (through its derivatives), never through the input layer
    d_x = xs[-1]
    d_y = min(all_ys) - 2.8
    out_y = drawn_positions[-1][0][0]
    ax.plot([xs[-1], d_x], [out_y, d_y + neuron_radius], color="#DD8452",
             linewidth=1.0, linestyle="--", alpha=0.8, zorder=1)
    circle = plt.Circle((d_x, d_y), neuron_radius, color="#DD8452", ec="black", zorder=3)
    ax.add_patch(circle)
    ax.text(d_x, d_y, "D", ha="center", va="center", fontsize=10, zorder=4, color="black")
    ax.text(d_x, d_y - neuron_spacing * 1.3, "Learnable scalar\n(not a network input)",
            ha="center", va="top", fontsize=9.5, color="#DD8452")

    ax.set_xlim(min(xs) - 1.5, max(xs) + 1.5)
    ax.set_ylim(d_y - 2.2, max(all_ys) + 2.0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("PINN architecture: (t, x, y) $\\rightarrow$ u(t, x, y), with D learned alongside", fontsize=13)

    fig.tight_layout()
    return fig, ax


if __name__ == "__main__":
    fig, ax = plot_architecture()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nn_architecture_part3.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved {out_path}")
