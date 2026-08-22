"""
Test / evaluation for the neuro-symbolic agent.

Plays one greedy episode 
"""
import os
import jax
import jax.numpy as jnp
import jaxatari
import numpy as np
import matplotlib.pyplot as plt

from navigation import plan, greedy_action
from mspacman_encoder import make_mspacman_encoder

MAZE_ID = 0
MAX_STEPS = 3000
SEED = 0
LAMBDA = 1.0           



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

def plot_curve(path="outputs/mspacman_returns.npy", out="outputs/curve.png"):
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

# --- Hand Crafted Danger --- 
THREAT = 50.0          # handcrafted per-ghost threat magnitude
RADIUS = 4             # handcrafted danger radius (cells)

def danger_field(ghost_cells, threat, shape=(40, 44), radius=RADIUS):
    """Fixed-radius danger, high near ghosts, fading to 0 at the radius edge.
 
    ghost_cells -- (n, 2) int ghost cells
    threat      -- (n,) per-ghost weight
    returns     -- (shape) danger field
    """
    xs = jnp.arange(shape[0])[:, None, None]
    ys = jnp.arange(shape[1])[None, :, None]
    gx = ghost_cells[:, 0][None, None, :]
    gy = ghost_cells[:, 1][None, None, :]
    dist = jnp.maximum(jnp.abs(xs - gx), jnp.abs(ys - gy))          # (W,H,n) box distance
    contrib = jnp.where(dist <= radius,
                        threat[None, None, :] * (1.0 - dist / (radius + 1)),
                        0.0)
    return contrib.max(axis=-1)                                     # strongest ghost per cell
 
 
def handcrafted_danger(snap, threat=THREAT):
    """Build danger_fn(obs) -> (H,W) using the handcrafted radius field."""
    def danger_fn(obs):
        gpos = obs.ghost_positions
        ggx = (gpos[:, 0] + 5) // 4
        ggy = (gpos[:, 1] + 3) // 4
        ghost_cells = jnp.stack([ggx, ggy], axis=-1)               # (n, 2)
        threats = jnp.ones(gpos.shape[0]) * threat
        return danger_field(ghost_cells, threats)
    return danger_fn

# --- Game Test ---
def run(env, enc, danger_fn, lam=LAMBDA, seed=SEED, max_steps=MAX_STEPS, render=True):
    """Play one greedy episode with the two-field planner. Returns (score, frames)."""
    maze, walkable = enc.maze, enc.walkable
    gx, gy = enc.gx, enc.gy
    snap, get_goal, features = enc.snap, enc.goal, enc.features
    
    @jax.jit
    def decide(obs, prev_dir):
        goals = get_goal(obs, walkable, gx, gy)
        Vn = plan(maze, goals, walkable)                       
        danger = danger_fn(obs)                                    # pluggable danger source
        return greedy_action(Vn, danger, maze, goals, obs.player_position, prev_dir, snap, lam)

    obs, state = env.reset(jax.random.PRNGKey(seed))
    prev_dir = jnp.int32(0)
    frames = [] if render else None

    for t in range(max_steps):
        action, prev_dir = decide(obs, prev_dir)
        obs, state, reward, done, info = env.step(state, action)
        if render:
            frames.append(np.asarray(env.render(state), dtype=np.uint8))
        if bool(done):
            break

    return int(state.score), frames

def main():
    env = jaxatari.make("mspacman")
    enc = make_mspacman_encoder(env, maze_id=MAZE_ID)
 
    # danger source - switch this for the test
    danger_fn = handcrafted_danger(enc.snap, threat=THREAT)
 
    score, frames = run(env, enc, danger_fn, lam=LAMBDA, render=True)
    print(f"[test] maze={MAZE_ID}  lambda={LAMBDA}  threat={THREAT}  radius={RADIUS}")
    print(f"[test] score={score}  frames={len(frames)}")
    #save_gif(frames, f"outputs/test_maze{MAZE_ID}.gif")
    #plot_curve()
 
if __name__ == "__main__":
    main()