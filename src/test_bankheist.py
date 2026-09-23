"""
Test / evaluation for the neuro-symbolic agent, Bank Heist version.

Plays one greedy episode. Mirrors the Ms. Pac-Man test.py structure exactly;
agent_MF.py (DangerNet, load_params) is imported and used UNCHANGED -- only
the game-specific plumbing (encoder, nav action mapping, danger-field
coordinate handling, score field) differs.

Key structural difference from the Ms. Pac-Man version, unavoidable given
how Bank Heist's map works: there is no fixed MAZE_ID to select ahead of
time (map_collision comes from live state, not a constructor argument), so
the encoder is built from a real reset() state rather than from env alone.
"""
import os
import jax
import jax.numpy as jnp
import jaxatari
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from navigation import plan, greedy_action
from encoders.bankhiest_encoder import make_bankheist_encoder


MAX_STEPS = 3000
SEED = 0
LAMBDA = 8.0
LARGE_COST = 1e6

TILE = 4  # confirmed real collision-box size (see bankheist_grid_encoder.py)

# --- Utility ---
def save_gif(frames, path, fps=30):
    try:
        import imageio.v2 as imageio
    except ImportError:
        print("[gif] pip install imageio to enable GIFs")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, frames, fps=fps)
    print(f"[gif] wrote {path}  ({len(frames)} frames)")

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

def danger_heatmap_frame(danger, player_px, enemy_px, snap, t,
                         enemy_active=None, auto_scale=True, danger_max=10.0):
    """Game-agnostic danger heatmap. Positions are PIXEL coords; `snap` maps them to cells."""
    d = np.asarray(danger)
    if auto_scale:
        vmin, vmax = float(d.min()), float(d.max())
        if vmax - vmin < 1e-6:                    # all-zero field (gate off) -> avoid degenerate scale
            vmax = vmin + 1.0
    else:
        vmin, vmax = 0.0, danger_max

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(d.T, origin="upper", cmap="hot", vmin=vmin, vmax=vmax, interpolation="nearest")
    plt.colorbar(im, ax=ax, label="danger")

    px, py = snap(jnp.asarray(player_px))
    ax.plot(int(px), int(py), "co", markersize=8)

    enemy_px = np.asarray(enemy_px)
    active = (np.ones(len(enemy_px), bool) if enemy_active is None
              else np.asarray(enemy_active) > 0)
    for (ex, ey), a in zip(enemy_px, active):
        if a:                                     # skip inactive police parked at (0,0)
            gx, gy = snap(jnp.array([ex, ey]))
            ax.plot(int(gx), int(gy), "b+", markersize=10, markeredgewidth=2)

    ax.set_title(f"danger t={t}  [{vmin:.2f}, {vmax:.2f}]")
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return frame


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
        pos = jnp.array([obs.player.x, obs.player.y])      # BankHeist player position
        action, d = greedy_action(V_nav, danger, maze, goals, pos, prev_dir, snap, lam)
        return action, d, V_nav, danger, goals

    obs, state = env.reset(jax.random.PRNGKey(seed))
    initial_map = state.map_collision
    prev_dir = jnp.int32(0)
    frames = [] if render else None
    heatmap = [] if render else None
    banks_seen = False
    recent = []

    for t in range(max_steps):
        action, prev_dir, V_nav, danger, goals = decide(obs, state, prev_dir)
        obs, state, reward, done, info = env.step(state, action)

        if not jnp.array_equal(state.map_collision, initial_map):
            print(f"[test] map changed at t={t}, encoder stale, stopping")
            break

        if int(goals.sum()) > 0:
            banks_seen = True
        elif banks_seen:
            print(f"[test] banks were present, now depleted at t={t}, stopping")
            break

        gxp, gyp = (int(c) for c in snap(jnp.array([obs.player.x, obs.player.y])))  # inside the loop
        recent = (recent + [(gxp, gyp)])[-40:]
        if len(recent) == 40 and len(set(recent)) <= 2 and t % 20 == 0:
            nbr = lambda f: [round(float(f[gxp, gyp-1]), 2), round(float(f[gxp+1, gyp]), 2),
                            round(float(f[gxp-1, gyp]), 2), round(float(f[gxp, gyp+1]), 2)]
            print(f"[stuck t={t}] cell=({gxp},{gyp})  nav={nbr(V_nav)}  dng={nbr(danger)}  (u,r,l,d)")

        if render:
            frames.append(np.asarray(env.render(state), dtype=np.uint8))
        if render and t % 2 == 0:              # every 2nd step, keep GIF size sane
            heatmap.append(danger_heatmap_frame(danger, jnp.array([obs.player.x, obs.player.y]),enc.enemy_pos(obs), snap, t, enemy_active=obs.enemies.active))
        if bool(done):
            break

    return int(state.money), frames, heatmap


# --- Danger fn ---
def make_net_danger_fn(env, enc, model_path):
    from agent_MF import DangerNet, load_params  # UNCHANGED -- reused as-is, not adapted
    net = DangerNet()
    obs, state = env.reset(jax.random.PRNGKey(0))
    template = net.init(jax.random.PRNGKey(0), enc.features(state, obs, enc.maze, enc.walkable))
    params = load_params(template, model_path)

    def danger_fn(obs, state):
        return net.apply(params, enc.features(state, obs, enc.maze, enc.walkable))
    return danger_fn


# --- Main ---
def main():
    env = jaxatari.make("bankheist")
    obs0, state0 = env.reset(jax.random.PRNGKey(SEED))
    enc = make_bankheist_encoder(state0)

    # danger source -- switch this for the test
    danger_fn = make_net_danger_fn(env, enc, "outputs/mspacman_best.msgpack")

    score, frames, heatmap = run(env, enc, danger_fn, lam=LAMBDA, seed=SEED)

    print(f"[test] money={score}  frames={len(frames)}")

    if frames:
        save_gif(frames, "gifs/bankheist_fixed.gif")
    if heatmap:
        save_gif(heatmap, f"gifs/heatmap_bankheist_fixed.gif", fps=15)

if __name__ == "__main__":
    main()