"""
Numerical solver + animation for the 2D heat equation:

    dT/dt = alpha * (d2T/dx2 + d2T/dy2)

Solved with an explicit finite-difference (FTCS) scheme on a rectangular
grid with Dirichlet boundary conditions. The result is animated as a
heatmap where blue = cold and red = hot.

Run directly to see a demo of a uniformly cold plate on [-1, 1] x [-1, 1]
that is suddenly heated on its boundary with
    T = 1 - y**2   on x = +-1
    T = x**2 - 1   on y = +-1
and watch the heat spread inward over time. The animation is also saved
as heat.mp4 in this folder:

    python heat.py
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def solve_heat_equation(
    T0,
    Lx=1.0,
    Ly=1.0,
    alpha=1.0,
    t_final=1.0,
    n_frames=120,
    boundary_temp=0.0,
    x0=0.0,
    y0=0.0,
):
    """Solve the 2D heat equation on a grid using explicit FTCS.

    Parameters
    ----------
    T0 : 2D array (ny, nx)
        Initial temperature field.
    Lx, Ly : float
        Physical size of the domain in x and y.
    alpha : float
        Thermal diffusivity.
    t_final : float
        Total simulated time.
    n_frames : int
        Number of frames to keep for the animation (subsampled from the
        internal, stability-constrained time steps).
    boundary_temp : float or callable
        Dirichlet BC imposed on all four edges. Either a fixed scalar, or a
        callable ``boundary_temp(X, Y)`` (X, Y being meshgrid coordinate
        arrays, shape (ny, nx)) returning the temperature at every point;
        only the values on the four edges of the returned array are used.
    x0, y0 : float
        Coordinate of the domain's lower-left corner (domain spans
        [x0, x0 + Lx] x [y0, y0 + Ly]).

    Returns
    -------
    x, y : 1D arrays of coordinates
    frames : 3D array (n_frames+1, ny, nx) of temperature snapshots
    times : 1D array of the time value for each frame
    """
    ny, nx = T0.shape
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
    x = np.linspace(x0, x0 + Lx, nx)
    y = np.linspace(y0, y0 + Ly, ny)

    if callable(boundary_temp):
        Xg, Yg = np.meshgrid(x, y)
        B = boundary_temp(Xg, Yg)
        bc_bottom, bc_top = B[0, :], B[-1, :]
        bc_left, bc_right = B[:, 0], B[:, -1]
    else:
        bc_bottom = bc_top = bc_left = bc_right = boundary_temp

    # Explicit scheme stability limit: dt <= 1 / (2*alpha*(1/dx^2 + 1/dy^2))
    dt_max = 1.0 / (2.0 * alpha * (1.0 / dx**2 + 1.0 / dy**2))
    dt = 0.5 * dt_max  # safety factor
    n_steps = max(1, int(np.ceil(t_final / dt)))
    dt = t_final / n_steps

    save_every = max(1, n_steps // n_frames)

    def apply_bc(T):
        T[0, :] = bc_bottom
        T[-1, :] = bc_top
        T[:, 0] = bc_left
        T[:, -1] = bc_right

    T = T0.copy()
    apply_bc(T)

    frames = [T.copy()]
    times = [0.0]

    rx = alpha * dt / dx**2
    ry = alpha * dt / dy**2

    for step in range(1, n_steps + 1):
        lap = (
            rx * (np.roll(T, -1, axis=1) - 2 * T + np.roll(T, 1, axis=1))
            + ry * (np.roll(T, -1, axis=0) - 2 * T + np.roll(T, 1, axis=0))
        )
        T = T + lap
        apply_bc(T)  # Enforce Dirichlet boundaries every step

        if step % save_every == 0 or step == n_steps:
            frames.append(T.copy())
            times.append(step * dt)

    return x, y, np.array(frames), np.array(times)


def save_frame(x, y, frame, t, save_path, vmin=None, vmax=None):
    """Save a single temperature frame as a static PNG heatmap."""
    style = open("plot_style.py").read()
    exec(style)

    fig, ax = plt.subplots(figsize=(6, 5))

    extent = [x.min(), x.max(), y.min(), y.max()]
    im = ax.imshow(
        frame,
        origin="lower",
        extent=extent,
        cmap="turbo",
        vmin=vmin,
        vmax=vmax,
        interpolation="bilinear",
        aspect="equal",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"T(x, y)   t = {t:.4f}")
    fig.colorbar(im, ax=ax, label="Temperature")

    fig.savefig(save_path)
    plt.close(fig)


def animate_heat(x, y, frames, times, interval_ms=40, save_path=None):
    """Animate temperature frames as a red/blue heatmap.

    x, y are the actual coordinate arrays, used to set the plot extent
    so the axes show real coordinates rather than array indices.
    """
    style = open("plot_style.py").read()
    exec(style)

    fig, ax = plt.subplots(figsize=(6, 5))

    vmin, vmax = frames.min(), frames.max()
    extent = [x.min(), x.max(), y.min(), y.max()]

    im = ax.imshow(
        frames[0],
        origin="lower",
        extent=extent,
        cmap="turbo",
        vmin=vmin,
        vmax=vmax,
        interpolation="bilinear",
        aspect="equal",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    title = ax.set_title(f"T(x, y)   t = {times[0]:.4f}")
    fig.colorbar(im, ax=ax, label="Temperature")

    def update(i):
        im.set_data(frames[i])
        title.set_text(f"T(x, y)   t = {times[i]:.4f}")
        return im, title

    anim = FuncAnimation(
        fig, update, frames=len(frames), interval=interval_ms, blit=False
    )

    if save_path is not None:
        anim.save(save_path, writer="ffmpeg", fps=1000 // interval_ms)

    return fig, anim


def _demo():
    nx, ny = 80, 80
    Lx, Ly = 2.0, 2.0
    x0, y0 = -1.0, -1.0

    # Uniformly cold plate, at the boundary's coldest value
    T0 = np.full((ny, nx), 0.0)

    def boundary_temp(X, Y):
        # X**2 - Y**2 + 1 reduces to 2 - y**2 on x = +-1
        # and to x**2 on y = +-1; shifted by +1 so T >= 0 everywhere
        return X**2 - Y**2 + 1

    x, y, frames, times = solve_heat_equation(
        T0,
        Lx=Lx,
        Ly=Ly,
        x0=x0,
        y0=y0,
        alpha=0.1,
        t_final=100.0,
        n_frames=500,
        boundary_temp=boundary_temp,
    )

    mp4_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "heat.mp4")
    fig, anim = animate_heat(x, y, frames, times, save_path=mp4_path)
    print(f"Saved animation to {mp4_path}")

    vmin, vmax = frames.min(), frames.max()
    here = os.path.dirname(os.path.abspath(__file__))
    first_path = os.path.join(here, "heat_first.png")
    last_path = os.path.join(here, "heat_last.png")
    save_frame(x, y, frames[0], times[0], first_path, vmin=vmin, vmax=vmax)
    save_frame(x, y, frames[-1], times[-1], last_path, vmin=vmin, vmax=vmax)
    print(f"Saved first frame to {first_path}")
    print(f"Saved last frame to {last_path}")

    plt.show()


if __name__ == "__main__":
    _demo()
