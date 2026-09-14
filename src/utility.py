import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

#JAXATARI
import jax
import jax.numpy as jnp
import jaxatari



# ----- Utility -----
def save_gif(frames, path, fps=30):
    try:
        import imageio.v2 as imageio
    except ImportError:
        print("[gif] pip install imageio to enable GIFs")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, frames, fps=fps)
    print(f"[gif] wrote {path}  ({len(frames)} frames)")

def plot_curve(path="outputs/returns.npy", out="outputs/curve.png"):
    try:
        r = np.load(path)
    except FileNotFoundError:
        print("[plot] no returns log found; skipping curve")
        return
    plt.plot(r, alpha=0.3, label="return")
    if len(r) >= 20:
        plt.plot(np.convolve(r, np.ones(20) / 20, mode="valid"), label="smoothed")
    plt.xlabel("epoch"); plt.ylabel("return"); plt.legend()
    plt.savefig(out)
    print(f"[plot] wrote {out}")

def save_danger_heatmap(danger, obs, snap, t, save_dir="outputs/debug_frames"):
    matplotlib.use("Agg")
    os.makedirs(save_dir, exist_ok=True)

    d = np.asarray(danger)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(d.T, origin="upper", cmap="hot")   # .T because maze is (x,y)
    plt.colorbar(im, ax=ax, label="danger")

    # mark pacman and ghosts
    gxp, gyp = snap(obs.player_position)
    ax.plot(int(gxp), int(gyp), "co", markersize=8, label="pac")   # cyan
    gpos = obs.ghost_positions
    for g in gpos:
        ggx = int((g[0] + 5) // 4); ggy = int((g[1] + 3) // 4)
        ax.plot(ggx, ggy, "b+", markersize=10)                     # blue + for ghosts
    ax.legend(); ax.set_title(f"danger field t={t}")
    plt.savefig(f"{save_dir}/danger_t{t:04d}.png", dpi=80)
    plt.close()

def danger_heatmap_frame(danger, obs, snap, t):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.asarray(danger)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(d.T, origin="upper", cmap="hot", vmin=0, vmax=10)   # fixed scale!
    plt.colorbar(im, ax=ax, label="danger")
    gxp, gyp = snap(obs.player_position)
    ax.plot(int(gxp), int(gyp), "co", markersize=8)
    for g in obs.ghost_positions:
        ggx = int((g[0] + 5) // 4); ggy = int((g[1] + 3) // 4)
        ax.plot(ggx, ggy, "b+", markersize=10)
    ax.set_title(f"danger t={t}")

    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3]  # drop alpha
    plt.close(fig)
    return frame

