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
LAMBDA = 4.0     
LARGE_COST = 1e6     

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

# --- Danger fn ---
def make_net_danger_fn(env, enc, model_path):
    from agent_MF import DangerNet, load_params 
    net = DangerNet()
    obs, _ = env.reset(jax.random.PRNGKey(0))   # you have obs already in main
    template = net.init(jax.random.PRNGKey(0), enc.features(obs, enc.maze, enc.walkable))
    params = load_params(template, model_path)
    def danger_fn(obs):
        return net.apply(params, enc.features(obs, enc.maze, enc.walkable))
    return danger_fn

# --- Game Test ---
def debug(walkable, maze, snap, obs, V_nav, danger, dist):
    gxp, gyp = snap(obs.player_position) 
    agent_mask = jnp.zeros(walkable.shape, bool).at[gxp,gyp].set(True)
    dist = plan(maze, agent_mask, walkable)

    gpos = obs.ghost_positions
    ggx = (gpos[:, 0] + 5) // 4
    ggy = (gpos[:, 1] + 3) // 4
    dist = [float(dist[int(ggx[i]), int(ggy[i])]) for i in range(gpos.shape[0])]
    nav_nbr = [float(V_nav[gxp, gyp-1]), float(V_nav[gxp+1, gyp]), float(V_nav[gxp-1, gyp]), float(V_nav[gxp, gyp+1])]
    dng_nbr = [float(danger[gxp, gyp-1]), float(danger[gxp+1, gyp]), float(danger[gxp-1, gyp]), float(danger[gxp, gyp+1])]
    # nearest ghost distance
    print(f"t={t}  ghost_nav_dist={[round(d,1) for d in dist]}")
    print(f"  nav   : {[round(n,1) for n in nav_nbr]}")
    print(f"  dng   : {[round(d,2) for d in dng_nbr]}")


def run(env, enc, danger_fn, lam=LAMBDA, seed=SEED, max_steps=MAX_STEPS, render=True):
    maze, walkable = enc.maze, enc.walkable
    gx, gy = enc.gx, enc.gy 
    snap, get_goal = enc.snap, enc.goal

    @jax.jit
    def decide(obs, prev_dir):
        goals = get_goal(obs, walkable, gx, gy)
        V_nav = plan(maze, goals, walkable)
        danger = danger_fn(obs)
        action, d = greedy_action(V_nav, danger, maze, goals, obs.player_position, prev_dir, snap, lam)
        return action, d, V_nav, danger      # return the fields for debugging

    obs, state = env.reset(jax.random.PRNGKey(seed))
    prev_dir = jnp.int32(0)
    frames = [] if render else None

    stuck_count = 0
    last_cell = None
    for t in range(max_steps):
        action, prev_dir, V_nav, danger = decide(obs, prev_dir)

        #stuck fix!
        cell = tuple(int(c) for c in snap(obs.player_position))
        if cell == last_cell:
            stuck_count += 1
        else:
            stuck_count = 0
        last_cell = cell
        if stuck_count > 3:
            goals = get_goal(obs, walkable, gx, gy)
            action, prev_dir = greedy_action(V_nav, jnp.zeros_like(danger), maze, goals, obs.player_position, prev_dir, snap, 0.0)
            stuck_count = 0

        obs, state, reward, done, info = env.step(state, action)

        if render:
            frames.append(np.asarray(env.render(state), dtype=np.uint8))
        if bool(done):
            break

    return int(state.score), frames

def main():
    #test environment and related game encoder head
    env = jaxatari.make("mspacman")
    enc = make_mspacman_encoder(env, maze_id=MAZE_ID)

    # danger source - switch this for the test
    #danger_fn = handcrafted_danger(enc.snap, threat=THREAT)
    danger_fn = make_net_danger_fn(env, enc, "outputs/mspacman_best.msgpack")
 
    score, frames = run(env, enc, danger_fn, lam=LAMBDA, render=True)
    print(f"[test] maze={MAZE_ID}  lambda={LAMBDA}  threat={THREAT}  radius={RADIUS}")
    print(f"[test] score={score}  frames={len(frames)}")
    save_gif(frames, f"outputs/test_new_reward_2.gif")
    plot_curve()
 
if __name__ == "__main__":
    main()