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

from navigation import plan, greedy_action
from encoders.bankhiest_encoder import make_bankheist_encoder


MAX_STEPS = 3000
SEED = 0
LAMBDA = 8.0
LARGE_COST = 1e6

TILE = 3  # confirmed real collision-box size (see bankheist_grid_encoder.py)

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


# --- Game Test ---
def run(env, enc, danger_fn, lam=LAMBDA, seed=SEED, max_steps=MAX_STEPS, render=True):
    maze, walkable = enc.maze, enc.walkable
    gx, gy = enc.gx, enc.gy
    snap, get_goal = enc.snap, enc.goal

    @jax.jit
    def decide(obs, state, prev_dir):
        goals = get_goal(obs, walkable, gx, gy, state)
        V_nav = plan(maze, goals, walkable)
        danger = danger_fn(obs, state)
        pos = jnp.array([obs.player.x, obs.player.y])      # BankHeist player position
        action, d = greedy_action(V_nav, danger, maze, goals, pos, prev_dir, snap, lam)
        return action, d, V_nav, danger, goals

    obs, state = env.reset(jax.random.PRNGKey(seed))
    initial_map = state.map_collision
    prev_dir = jnp.int32(0)
    frames = [] if render else None
    #heatmap = [] if render else None
    banks_seen = False

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

        if render:
            frames.append(np.asarray(env.render(state), dtype=np.uint8))
            
        if bool(done):
            break

    return int(state.money), frames


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

    print("walkable shape:", enc.walkable.shape, " maze shape:", enc.maze.shape)
    print("features shape:", enc.features(state0, obs0, enc.maze, enc.walkable).shape)

    # danger source -- switch this for the test
    danger_fn = make_net_danger_fn(env, enc, "outputs/weights/mspacman_v4.msgpack")

    score, frames = run(env, enc, danger_fn, lam=LAMBDA, seed=SEED)

    print(f"[test] money={score}  frames={len(frames)}")
    save_gif(frames, "gifs/bankheist_1.gif")

if __name__ == "__main__":
    main()