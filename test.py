"""
Test / evaluation for the neuro-symbolic agent. Loads a trained DangerNet, runs it greedily in the game, prints the score, and saves a GIF.
"""
import os
import jax
import jax.numpy as jnp
import jaxatari
import numpy as np
import matplotlib
matplotlib.use("Agg")                     # no display under WSL; write PNG only
import matplotlib.pyplot as plt

from navigation import plan
from agent import DangerNet, load_params, LARGE_COST, DIR_TO_ACTION
from mspacman_encoder import make_mspacman_encoder

MODEL_PATH = "outputs/mspacman_best.msgpack"
MAZE_ID = 0                             
MAX_STEPS = 3000
SEED = 0


# --- Hand Made Danger --- 
def danger_field(ghost_cells, threat, shape=(40, 44), radius=4):
    """Fixed-radius danger, high near ghosts, 0 beyond. threat: (n,) per-ghost weight."""
    xs = jnp.arange(shape[0])[:, None, None]      # (W,1,1)
    ys = jnp.arange(shape[1])[None, :, None]      # (1,H,1)
    gx = ghost_cells[:, 0][None, None, :]         # (1,1,n)
    gy = ghost_cells[:, 1][None, None, :]
    dist = jnp.maximum(jnp.abs(xs - gx), jnp.abs(ys - gy))          # (W,H,n) box distance
    contrib = jnp.where(dist <= radius, threat[None, None, :] * (1.0 - dist / (radius + 1)), 0.0)
    return contrib.max(axis=-1)


def run(env, enc, params, seed=SEED, max_steps=MAX_STEPS, render=True):
    """Play one greedy episode. Returns (score, frames)."""
    maze, walkable = enc.maze, enc.walkable
    gx, gy = enc.gx, enc.gy
    snap, get_goal, features = enc.snap, enc.goal, enc.features
    net = DangerNet()

    @jax.jit
    def decide(obs, prev_dir):
        goals = get_goal(obs, walkable, gx, gy)
        V_nav = plan(maze, goals, walkable, jnp.zeros(walkable.shape))  # danger=0 -> pure nav
        ghost_cells = jnp.stack(snap(obs.ghost_positions.T), axis=-1)   # (n,2) ghost cells
        threat = jnp.ones(obs.ghost_positions.shape[0]) * 50                # hand-set: all ghosts = 1
        danger = danger_field(ghost_cells, threat)
        return greedy_action(V_nav, danger, maze, goals, obs.player_position, prev_dir, snap)

    obs, state = env.reset(jax.random.PRNGKey(seed))
    prev_dir = jnp.int32(0)
    frames = [] if render else None

    for t in range(max_steps):
        action, prev_dir = decide(obs, prev_dir)

        if t % 100 == 0:
            danger = net.apply(params, features(obs, maze, walkable))
            field_mean, ghost_vals = ghost_diagnostic(obs, danger, walkable, snap)
            print(f"t={t:4d}  field_mean={field_mean:5.2f}  "
                  f"max={float(danger.max()):5.2f}  ghost_cells={ghost_vals}")

        obs, state, reward, done, info = env.step(state, action)
        if render:
            frames.append(np.asarray(env.render(state), dtype=np.uint8))
        if bool(done):
            break

    return int(state.score), frames


def save_gif(frames, path, fps=30):
    try:
        import imageio.v2 as imageio
    except ImportError:
        print("[gif] pip install imageio to enable GIFs")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, frames, fps=fps)
    print(f"[gif] wrote {path}  ({len(frames)} frames)")


def main():
    env = jaxatari.make("mspacman")
    enc = make_mspacman_encoder(env, maze_id=MAZE_ID)

    obs, _ = env.reset(jax.random.PRNGKey(0))
    assert enc.maze.shape[-1] == 4          # DOF
    assert enc.walkable.shape == enc.maze.shape[:2]
    assert callable(enc.snap) and callable(enc.goal) and callable(enc.features)
    print("encoder contract OK")
    return
    net = DangerNet()
    template = net.init(jax.random.PRNGKey(0), enc.features(obs, enc.maze, enc.walkable))
    params = load_params(template, MODEL_PATH)
    print(f"[test] loaded {MODEL_PATH}, maze {MAZE_ID}")

    score, frames = run(env, enc, params, render=True)
    print(f"[test] score={score}  frames={len(frames)}")
    save_gif(frames, f"outputs/test_maze{MAZE_ID}.gif")

    try:
        r = np.load("outputs/mspacman_returns.npy")
        plt.plot(r, alpha=0.3, label="return")
        if len(r) >= 20:
            plt.plot(np.convolve(r, np.ones(20) / 20, mode="valid"), label="smoothed")
        plt.xlabel("epoch"); plt.ylabel("return"); plt.legend()
        plt.savefig("outputs/curve.png")
        print("[plot] wrote outputs/curve.png")
    except FileNotFoundError:
        print("[plot] no returns log found; skipping curve")


if __name__ == "__main__":
    main()