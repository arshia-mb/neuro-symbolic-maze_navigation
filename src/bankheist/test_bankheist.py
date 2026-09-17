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

from .bh_nav import plan, greedy_action
from .grid_encoder import make_bankheist_encoder

from ..utility import save_gif


MAX_STEPS = 3000
SEED = 0
LAMBDA = 8.0
LARGE_COST = 1e6

TILE = 8  # confirmed real collision-box size (see bankheist_grid_encoder.py)


# --- Hand Crafted Danger ---
THREAT = 50.0
RADIUS = 4


def danger_field(enemy_cells, threat, shape, radius=RADIUS):
    """Fixed-radius danger, high near police cars, fading to 0 at the radius edge.

    enemy_cells -- (n, 2) int police-car cells
    threat      -- (n,) per-enemy weight
    returns     -- (shape) danger field
    """
    xs = jnp.arange(shape[0])[:, None, None]
    ys = jnp.arange(shape[1])[None, :, None]
    gx = enemy_cells[:, 0][None, None, :]
    gy = enemy_cells[:, 1][None, None, :]
    dist = jnp.maximum(jnp.abs(xs - gx), jnp.abs(ys - gy))
    contrib = jnp.where(dist <= radius,
                         threat[None, None, :] * (1.0 - dist / (radius + 1)),
                         0.0)
    return contrib.max(axis=-1)


def handcrafted_danger(snap, walkable_shape, threat=THREAT):
    """Build danger_fn(obs, state) -> (W,H) using the handcrafted radius
    field. Uses enc.snap() for pixel->tile conversion rather than a
    hardcoded offset formula, since Bank Heist's grid is a uniform TILE=8
    grid (no irregular offsets like Ms. Pac-Man's pellet grid had)."""
    def danger_fn(obs, state):
        enemy_gx, enemy_gy = jax.vmap(snap)(jnp.stack([obs.enemies.x, obs.enemies.y], axis=-1))
        enemy_cells = jnp.stack([enemy_gx, enemy_gy], axis=-1)
        active = obs.enemies.active > 0
        threats = jnp.where(active, threat, 0.0)  # inactive police contribute zero danger
        return danger_field(enemy_cells, threats, walkable_shape)
    return danger_fn


# ----- Debug -----
def debug(walkable, maze, snap, obs, state, env, V_nav, danger, t, save_dir="outputs/debug_frames"):
    pos = jnp.array([obs.player.x, obs.player.y])
    gxp, gyp = snap(pos)
    agent_mask = jnp.zeros(walkable.shape, bool).at[gxp, gyp].set(True)
    dist = plan(maze, agent_mask, walkable)

    enemy_gx, enemy_gy = jax.vmap(snap)(jnp.stack([obs.enemies.x, obs.enemies.y], axis=-1))
    dist_to_enemies = [float(dist[int(enemy_gx[i]), int(enemy_gy[i])]) for i in range(obs.enemies.x.shape[0])]
    nav_nbr = [float(V_nav[gxp, gyp - 1]), float(V_nav[gxp + 1, gyp]),
               float(V_nav[gxp - 1, gyp]), float(V_nav[gxp, gyp + 1])]
    dng_nbr = [float(danger[gxp, gyp - 1]), float(danger[gxp + 1, gyp]),
               float(danger[gxp - 1, gyp]), float(danger[gxp, gyp + 1])]

    print(f"t={t}  enemy_nav_dist={[round(d, 1) for d in dist_to_enemies]}")
    print(f"  nav   : {[round(n, 1) for n in nav_nbr]}")
    print(f"  dng   : {[round(d, 2) for d in dng_nbr]}")

    os.makedirs(save_dir, exist_ok=True)
    frame = np.asarray(env.render(state), dtype=np.uint8)
    try:
        import imageio.v2 as imageio
        imageio.imwrite(f"{save_dir}/t{t:04d}.png", frame)
    except ImportError:
        pass


def audit_features(enc, state, obs, maze, walkable, snap):
    f = enc.features(state, obs, maze, walkable)
    print("feature shape:", f.shape, "  nan:", bool(jnp.isnan(f).any()), "  inf:", bool(jnp.isinf(f).any()))

    # Matches bankheist_grid_encoder.py's 8-channel schema, which mirrors
    # mspacman_encoder.extract_features exactly (see that file's docstring
    # for what's confirmed vs. assumed per channel -- ch6/7 in particular
    # rely on an unverified interpretation of obs.enemies.orientation).
    names = ["DOF up", "DOF right", "DOF left", "DOF down",
             "dist-to-player", "dist-to-danger", "enemy vx", "enemy vy"]
    for c in range(f.shape[-1]):
        ch = f[..., c]
        print(f"ch{c} {names[c]:14s}: min {float(ch.min()):8.2f}  max {float(ch.max()):8.2f} "
              f"mean {float(ch.mean()):6.2f}  nonzero {int((ch != 0).sum())}")

    enemy_gx, enemy_gy = jax.vmap(snap)(jnp.stack([obs.enemies.x, obs.enemies.y], axis=-1))
    print("\n-- enemy cross-check --")
    print("enemy cells:", [(int(enemy_gx[i]), int(enemy_gy[i])) for i in range(obs.enemies.x.shape[0])])
    dist_to_danger = f[..., 5]
    for i in range(obs.enemies.x.shape[0]):
        gx_, gy_ = int(enemy_gx[i]), int(enemy_gy[i])
        active = bool(obs.enemies.active[i] > 0)
        print(f"  enemy {i} at ({gx_},{gy_}) active={active}: "
              f"dist_to_danger_here={float(dist_to_danger[gx_, gy_]):.2f} "
              f"(expect ~0.0 if active, since this cell IS a danger source)")


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
        pos = jnp.array([obs.player.x, obs.player.y])
        action, d = greedy_action(V_nav, danger, maze, goals, pos, prev_dir, snap, lam)
        return action, d, V_nav, danger

    obs, state = env.reset(jax.random.PRNGKey(seed))
    prev_dir = jnp.int32(0)
    frames = [] if render else None
    heatmap = [] if render else None

    for t in range(max_steps):
        action, prev_dir, V_nav, danger = decide(obs, state, prev_dir)

        obs, state, reward, done, info = env.step(state, action)  # real 5-value return

        if render:
            frames.append(np.asarray(env.render(state), dtype=np.uint8))
        if bool(done):
            break

    return int(state.money), frames, heatmap  # confirmed real field: state.money (not state.score)


# --- Danger fn ---
def make_net_danger_fn(env, enc, model_path):
    from src.agent_MF import DangerNet, load_params  # UNCHANGED -- reused as-is, not adapted
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

    # Encoder is built from a live reset state (map_collision lives on
    # state, not passed as a constructor arg like Ms. Pac-Man's maze_id) --
    # this is the one unavoidable structural difference from the reference.
    obs0, state0 = env.reset(jax.random.PRNGKey(SEED))
    enc = make_bankheist_encoder(state0)

    # danger source -- switch this for the test
    # danger_fn = handcrafted_danger(enc.snap, enc.walkable.shape, threat=THREAT)
    danger_fn = make_net_danger_fn(env, enc, "outputs/weights/bankheist_v1.msgpack")

    score, frames, heatmap = run(env, enc, danger_fn, lam=LAMBDA, render=True)
    print(f"[test] lambda={LAMBDA}")
    print(f"[test] score={score}  frames={len(frames)}")
    save_gif(frames, "outputs/bankheist/bh_debug.gif")
    save_gif(heatmap, "outputs/bankheist/bh_heatmap.gif", fps=15)


if __name__ == "__main__":
    main()