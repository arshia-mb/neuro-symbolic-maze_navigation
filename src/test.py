"""
Test / evaluation for the neuro-symbolic agent.

Plays one greedy episode 
"""
import os
import jax
import jax.numpy as jnp
import jaxatari
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from navigation import plan, greedy_action
from encoders.mspacman_encoder import make_mspacman_encoder
from encoders.pacman_encoder import make_pacman_encoder

MAX_STEPS = 3000
SEED = 0
LAMBDA = 8.0   
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

# --- Hand Crafted Danger --- 
THREAT = 50.0          # handcrafted per-ghost threat magnitude
RADIUS = 4             # handcrafted danger radius (cells)
def danger_field(ghost_cells, threat, shape, radius=RADIUS):
    """Fixed-radius danger, high near enemy, fading to 0 at the radius edge.
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

def handcrafted_danger(snap, shape, threat=THREAT):
    """Build danger_fn(obs) -> (H,W) using the handcrafted radius field.
    """
    def danger_fn(obs, state=None):
        gpos = obs.ghost_positions
        ggx = (gpos[:, 0] + 5) // 4
        ggy = (gpos[:, 1] + 3) // 4
        ghost_cells = jnp.stack([ggx, ggy], axis=-1)
        threats = jnp.ones(gpos.shape[0]) * threat
        return danger_field(ghost_cells, threats, shape=shape)
    return danger_fn

# ----- Debug -----
def debug(walkable, maze, snap, obs, state, env, V_nav, danger, t, save_dir="outputs/debug_frames"):
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

    #save actual game frame
    os.makedirs(save_dir, exist_ok=True)
    frame = np.asarray(env.render(state), dtype=np.uint8)
    try:
        import imageio.v2 as imageio
        imageio.imwrite(f"{save_dir}/t{t:04d}.png", frame)
    except ImportError:
        pass

# --- Game Test ---   
def run(env, enc, danger_fn, lam=LAMBDA, seed=SEED, max_steps=MAX_STEPS, render=True):
    maze, walkable = enc.maze, enc.walkable
    gx, gy = enc.gx, enc.gy 
    snap, get_goal = enc.snap, enc.goal

    @jax.jit
    def decide(obs, state, prev_dir):
        goals = get_goal(obs, walkable, gx, gy)
        V_nav = plan(maze, goals, walkable)
        danger = danger_fn(obs, state)
        action, d = greedy_action(V_nav, danger, maze, goals, obs.player_position, prev_dir, snap, lam)
        return action, d, V_nav, danger      # return the fields for debugging

    obs, state = env.reset(jax.random.PRNGKey(seed))
    prev_dir = jnp.int32(0)
    frames = [] if render else None
    heatmap = [] if render else None
    print(f"[debug] initial state: score={state.score}  lives={state.lives}")

    for t in range(max_steps):
        action, prev_dir, V_nav, danger = decide(obs, state, prev_dir)
        obs, state, reward, done, info = env.step(state, action)
        if render:
            frames.append(np.asarray(env.render(state), dtype=np.uint8))
        if render and t % 2 == 0:              # every 2nd step, keep GIF size sane
            heatmap.append(danger_heatmap_frame(danger, obs, snap, t))
        if bool(done):
            break

    return int(state.score), frames, heatmap

def run_batch(env, enc, danger_fn, seeds, lam=LAMBDA, max_steps=MAX_STEPS):
    """Vectorized eval over many seeds. Returns per-seed final score. No rendering."""
    maze, walkable = enc.maze, enc.walkable
    gx, gy = enc.gx, enc.gy
    snap, get_goal = enc.snap, enc.goal

    @jax.jit
    def episode(seed):
        obs, state = env.reset(jax.random.PRNGKey(seed))
        prev_dir = jnp.int32(0)
        init_carry = (obs, state, prev_dir)

        def step(carry, _):
            obs, state, prev_dir = carry
            goals = get_goal(obs, walkable, gx, gy)
            V_nav = plan(maze, goals, walkable)
            danger = danger_fn(obs, state)
            action, prev_dir = greedy_action(V_nav, danger, maze, goals,
                                              obs.player_position, prev_dir, snap, lam)
            obs, state, reward, done, info = env.step(state, action)
            return (obs, state, prev_dir), (state.score, done)

        (obs, state, _), (scores, dones) = jax.lax.scan(step, init_carry, xs=None, length=max_steps)

        # score at the step the episode actually ended (first done), else the last step
        first_done = jnp.argmax(dones)                     # index of first True (0 if never done)
        ended = jnp.any(dones)
        final_idx = jnp.where(ended, first_done, max_steps - 1)
        return scores[final_idx]

    return jax.vmap(episode)(seeds)     # (n_seeds,) array of final scores

# --- Danger fn ---
def make_net_danger_fn(env, enc, model_path):
    from agent_MF import DangerNet, load_params 
    net = DangerNet()
    obs, state = env.reset(jax.random.PRNGKey(0))   # you have obs already in main
    template = net.init(jax.random.PRNGKey(0), enc.features(state, obs, enc.maze, enc.walkable))
    params = load_params(template, model_path)
    def danger_fn(obs, state):
        return net.apply(params, enc.features(state, obs, enc.maze, enc.walkable))
    return danger_fn

# --- Main ---
def main(maze_id=0, mode="score", seed=0, n_seeds=10):
    env = jaxatari.make("pacman")
    env.consts = env.consts.replace(RESET_LEVEL=1 + 2 * maze_id)
    enc = make_pacman_encoder(env, maze_id=maze_id)

    #danger_fn = handcrafted_danger(enc.snap, shape=enc.walkable.shape, threat=THREAT)
    danger_fn = make_net_danger_fn(env, enc, "outputs/weights/mspacman_v4.msgpack")

    if mode == "render":
        score, frames, heatmap = run(env, enc, danger_fn, lam=LAMBDA, seed=seed)
        print(f"[test] maze={maze_id} seed={seed} score={score}")
        save_gif(frames, f"gifs/pacman_{maze_id}_{seed}.gif")
        if heatmap:
            save_gif(heatmap, f"gifs/heatmap_pacman_{maze_id}_{seed}.gif", fps=15)
        return score

    elif mode == "score":
        seeds = jnp.arange(n_seeds)
        scores = run_batch(env, enc, danger_fn, seeds)
        print(f"[test] maze={maze_id}  mean={float(scores.mean()):.1f}  std={float(scores.std()):.1f}")
        return scores

if __name__ == "__main__":
    # fast batched scoring across mazes
    for maze_id in range(4):
        try:
            main(maze_id=maze_id, mode="score", n_seeds=10)
        except Exception as e:
            print(f"Error on maze {maze_id}: {e}")

    # one rendered demo (pick a maze/seed you care about for the report)
    main(maze_id=0, mode="render", seed=0)