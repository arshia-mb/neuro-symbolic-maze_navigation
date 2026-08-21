"""
Test / evaluation for the neuro-symbolic agent. Loads a trained DangerNet, runs it greedily in the game, prints the score, and saves a GIF.
"""

import jax
import jax.numpy as jnp
import jaxatari
import numpy as np
import os
import numpy as np, matplotlib.pyplot as plt
from navigation import plan 
from agent import DangerNet, load_params, LARGE_COST, DIR_TO_ACTION
from mspacman_encoder import make_mspacman_encoder

MODEL_PATH = "outputs/mspacman_best.msgpack"   # the good checkpoint (score ~900)
MAZE_ID = 0                                    # change to 1/2/3 to test other layouts
MAX_STEPS = 3000
SEED = 0


def greedy_action(V, maze, goal_mask, pos, prev_dir, snap):
    """Deterministic descent: step to the legal neighbour with lowest value."""
    gx, gy = snap(pos)
    V_nbr = jnp.array([V[gx, gy - 1], V[gx + 1, gy], V[gx - 1, gy], V[gx, gy + 1]])
    legal = maze[gx, gy]
    V_nbr = jnp.where(legal, V_nbr, LARGE_COST)
    tie = jnp.zeros(4).at[prev_dir].add(-1e-3)          # break ties toward heading
    d = jnp.argmin(V_nbr + tie).astype(jnp.int32)
    d = jnp.where(goal_mask[gx, gy], prev_dir, d).astype(jnp.int32)   # coast on goal cell
    return d + DIR_TO_ACTION, d

def compute_danger(obs, maze, walkable, params, net, features):
    return net.apply(params, features(obs, maze, walkable))

def run(env, enc, params, seed=SEED, max_steps=MAX_STEPS, render=True):
    """Play one greedy episode. Returns (score, frames)."""
    maze, walkable = enc.maze, enc.walkable
    gx, gy = enc.gx, enc.gy
    snap, get_goal, features = enc.snap, enc.goal, enc.features
 
    net = DangerNet()
 
    obs, state = env.reset(jax.random.PRNGKey(seed))
    prev_dir = jnp.int32(0)
    frames = [] if render else None

    @jax.jit
    def decide(obs, prev_dir):
        danger = net.apply(params, features(obs, maze, walkable))
        sc = jax.nn.softplus(danger)
        goals = get_goal(obs, walkable, gx, gy)
        V = plan(maze, goals, walkable, danger)
        return greedy_action(V, maze, goals, obs.player_position, prev_dir, snap)
 
    for t in range(max_steps):
        action, prev_dir = decide(obs, prev_dir)
        if t%100==0:
            danger = compute_danger(obs, maze, walkable, params, net, features)
            sc = jax.nn.softplus(danger)
            print("danger:", float(danger.min()), float(danger.mean()), float(danger.max()),"| softplus:", float(sc.min()), float(sc.mean()), float(sc.max()))
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
 
    # load params: build a template of the right shape, then fill from disk
    obs, _ = env.reset(jax.random.PRNGKey(0))
    net = DangerNet()
    template = net.init(jax.random.PRNGKey(0), enc.features(obs, enc.maze, enc.walkable))
    params = load_params(template, MODEL_PATH)
    print(f"[test] loaded {MODEL_PATH}, maze {MAZE_ID}")
 
    score, frames = run(env, enc, params, render=True)
    print(f"[test] score={score}  frames={len(frames)}")
    save_gif(frames, f"outputs/test_maze{MAZE_ID}.gif")

    r = np.load("outputs/mspacman_returns.npy")
    plt.plot(r, alpha=0.3, label="return")
    plt.plot(np.convolve(r, np.ones(20)/20, mode="valid"), label="smoothed")
    plt.xlabel("epoch"); plt.ylabel("return"); plt.legend(); plt.savefig("outputs/curve.png")
 
 
if __name__ == "__main__":
    main()