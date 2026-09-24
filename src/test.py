"""
Test / evaluation for the neuro-symbolic agent.
 
All game-specific details come from the GameEncoder, so this file is game-agnostic.
 
Modes:
  render -- play one greedy episode on one maze, save a game GIF and a danger-heatmap GIF
  score  -- batched evaluation (jit + vmap over seeds) on every selected maze, no rendering;
            prints per-maze and overall statistics. Use this mode for reported numbers.
 
Both modes use the same decision function (`make_decide`), so rendered runs and
batched scores always evaluate the same policy.
 
Danger sources:
  net         -- learned DangerNet (default)
  handcrafted -- fixed-radius danger around every dangerous enemy (baseline)
  none        -- zero danger, i.e. navigation only (baseline)
 
Usage:
  python src/test.py --mode render --maze 0 --seed 0
  python src/test.py --mode render --game bankheist --danger handcrafted
  python src/test.py --mode score                          # all mazes, 10 seeds each
  python src/test.py --mode score --danger none --n-seeds 20
  python src/test.py --mode score --game pacman --mazes 0 2
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "default")   # fast allocator for evaluation
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.6")

import argparse

import matplotlib
matplotlib.use("Agg")  # headless rendering (WSL / no display)
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import jaxatari
import numpy as np


from navigation import plan, greedy_action
from agent_MF import DangerNet, load_params, danger_cost, LAMBDA, DANGER_MAX
from encoders.mspacman_encoder import make_mspacman_encoder
from encoders.pacman_encoder import make_pacman_encoder
from encoders.bankheist_encoder import make_bankheist_encoder

ENCODERS = {
    "mspacman": make_mspacman_encoder, 
    "pacman": make_pacman_encoder,
    "bankheist": make_bankheist_encoder
    }
MAZE_GAMES = {"mspacman", "pacman"}  #games whose start maze is chosen via RESET_LEVEL
N_MAZES = {"mspacman": 3, "pacman": 4, "bankheist": 1}   # BankHeist always starts on city 0

WEIGHTS = "models/mspacman_v6.msgpack"
OUT_DIR = "outputs/gifs"

MAX_STEPS = 3000
HEATMAP_EVERY = 4   #render a heatmap frame every N steps
HEATMAP_DPI = 60    #small figures keep render-mode memory low
DEBUG = False       #print neighbor values when the agent looks stuck - or other debug purposes
STUCK_WINDOW = 40   # in frames; the agent needs several frames per cell


# ----- Environment -----
def make_env(game, maze_id=0):
    """Fresh env per maze: jitted reset/step are cached on the env object, so reusing one env across mazes could silently keep the old maze."""
    env = jaxatari.make(game)
    if game in MAZE_GAMES:
        env.consts = env.consts.replace(RESET_LEVEL=1 + 2 * maze_id)   # start on a level using maze_id
    return env

# --- Danger source ---
def make_net_danger_fn(env, enc, model_path):
    """Learned Danger: the net applies the danger based on enc features"""
    net = DangerNet()
    obs, state = env.reset(jax.random.PRNGKey(0))
    template = net.init(jax.random.PRNGKey(0), enc.features(state, obs, enc.maze, enc.walkable))
    params = load_params(template, model_path)

    def danger_fn(obs, state):
        return net.apply(params, enc.features(state, obs, enc.maze, enc.walkable))
    return danger_fn

#handcrafted danger
THREAT = DANGER_MAX
RADIUS = 4 

def danger_field(enemy_cells, threat, shape, radius=RADIUS):
    """Box-distance danger around each enemy, fading linearly to 0 at `radius`."""
    xs = jnp.arange(shape[0])[:, None, None]
    ys = jnp.arange(shape[1])[None, :, None]
    ex = enemy_cells[:, 0][None, None, :]
    ey = enemy_cells[:, 1][None, None, :]
    dist = jnp.maximum(jnp.abs(xs - ex), jnp.abs(ys - ey))
    contrib = jnp.where(dist <= radius, threat[None, None, :] * (1.0 - dist / (radius + 1)), 0.0)
    return contrib.max(axis=-1) # strongest enemy per cell 

def make_handcrafted_danger_fn(enc, threat=THREAT, radius=RADIUS):
    """Handcrafted baseline: fixed-radius danger around every dangerous enemy."""
    shape = enc.walkable.shape
 
    def danger_fn(obs, state):
        pos = enc.enemy_pos(obs)
        ex, ey = jax.vmap(enc.snap)(pos)
        cells = jnp.stack([ex, ey], axis=-1)
        threats = jnp.where(enc.enemy_active(obs, state), threat, 0.0)
        return danger_field(cells, threats, shape, radius)
    return danger_fn

#navigation baseline
def make_zero_danger_fn(enc):
    """Navigation-only baseline."""
    zeros = jnp.zeros(enc.walkable.shape)
    return lambda obs, state: zeros

def make_danger_fn(kind, env, enc, weights=WEIGHTS):
    """Build the danger source selected on the command line."""
    if kind == "net":
        return make_net_danger_fn(env, enc, weights)
    if kind == "handcrafted":
        return make_handcrafted_danger_fn(enc)
    return make_zero_danger_fn(enc)

# --- Decision rule ---
def make_decide(enc, danger_fn, lam=LAMBDA):
    """Descend V_nav + lam * danger_cost(danger) over the four neighbors."""
    maze, walkable, gx, gy = enc.maze, enc.walkable, enc.gx, enc.gy
 
    def decide(obs, state, prev_dir):
        goals = enc.goal(obs, walkable, gx, gy)
        V = plan(maze, goals, walkable)
        danger = danger_fn(obs, state)
        action, d = greedy_action(V, danger_cost(danger), maze, goals, enc.player_pos(obs), prev_dir, enc.snap, lam)
        return action, d, V, danger
    return decide

# --- Game Tests ---   
def run(env, enc, danger_fn, lam=LAMBDA, seed=0, max_steps=MAX_STEPS, debug=DEBUG):
    """One greedy episode with rendering"""
    decide = jax.jit(make_decide(enc, danger_fn, lam))

    obs, state = env.reset(jax.random.PRNGKey(seed)) 
    prev_dir = jnp.int32(0)
    frames, heatmap, recent = [], [], []   

    #print(f"[debug] initial state: score={state.score}  lives={state.lives}")

    for t in range(max_steps):
        action, prev_dir, V, danger = decide(obs, state, prev_dir)
        obs, state, reward, done, info = env.step(state, action)

        if debug:
            cell = tuple(int(c) for c in enc.snap(enc.player_pos(obs)))
            recent = (recent + [cell])[-STUCK_WINDOW:]
            if len(recent) == STUCK_WINDOW and len(set(recent)) <= 2 and t % 20 == 0:
                print(f"[stuck t={t}] cell={cell} nav={neighbors(V, *cell)}  dng={neighbors(danger, *cell)}  (u,r,l,d)")

        frames.append(np.asarray(env.render(state), dtype=np.uint8))
        if t % HEATMAP_EVERY == 0:            
             heatmap.append(danger_heatmap_frame(danger, enc.player_pos(obs), enc.enemy_pos(obs), enc.snap, t, enemy_active=enc.enemy_active(obs, state)))

        if bool(done):
            break
        if not bool(enc.valid(state)):
            print(f"[run] encoder no longer matches the game at t={t} (map/maze changed), stopping")
            break

    return int(enc.score(state)), frames, heatmap

def run_batch(env, enc, danger_fn, seeds, lam=LAMBDA, max_steps=MAX_STEPS):
    """Batched greedy episodes over `seeds` (no rendering). Returns final score per seed."""
    decide = make_decide(enc, danger_fn, lam)

    def episode(seed):
        obs, state = env.reset(jax.random.PRNGKey(seed))

        def step(carry, _):
            obs, state, prev_dir, alive, score = carry
            action, prev_dir, _, _ = decide(obs, state, prev_dir)
            obs, state, _, done, _ = env.step(state, action)
            score = jnp.where(alive, enc.score(state).astype(jnp.float32), score)
            alive = alive & ~done & enc.valid(state)
            return (obs, state, prev_dir, alive, score), None
            
        init = (obs, state, jnp.int32(0), jnp.bool_(True), enc.score(state).astype(jnp.float32))
        (_, _, _, _, score), _ = jax.lax.scan(step, init, xs=None, length=max_steps)
        return score

    return jax.jit(jax.vmap(episode))(seeds)    # (n_seeds,) array of final scores

def score_mazes(game, mazes, n_seeds, danger, weights=WEIGHTS, max_steps=MAX_STEPS):
    """Run `n_seeds` batched episodes on every maze in `mazes`. 
    Prints per-maze and overall statistics; returns all scores."""
    all_scores = []
    for m in mazes:
        env = make_env(game, m)
        enc = ENCODERS[game](env, maze_id=m)
 
        # guard the assumption that the start layout doesn't depend on the seed
        bad = [s for s in range(n_seeds)
               if not bool(enc.valid(env.reset(jax.random.PRNGKey(s))[1]))]
        if bad:
            print(f"[warn] maze {m}: encoder doesn't match the start layout for seeds {bad}")
 
        danger_fn = make_danger_fn(danger, env, enc, weights)
        scores = np.asarray(run_batch(env, enc, danger_fn, jnp.arange(n_seeds), max_steps=max_steps))
        print(f"[score] {game} maze={m} danger={danger}  "
              f"mean={scores.mean():.1f}  std={scores.std():.1f}  "
              f"min={scores.min():.0f}  max={scores.max():.0f}  (n={n_seeds})")
        all_scores.append(scores)
 
    all_scores = np.concatenate(all_scores)
    print(f"[score] {game} ALL mazes={list(mazes)} danger={danger} mean={all_scores.mean():.1f}  std={all_scores.std():.1f}  (n={len(all_scores)})")
    return all_scores

# ----- Utility -----
def neighbors(field, x, y):
    """Field values at the four neighbors of (x, y), in [up, right, left, down] order."""
    vals = (field[x, y - 1], field[x + 1, y], field[x - 1, y], field[x, y + 1])
    return [round(float(v), 2) for v in vals]

def save_gif(frames, path, fps=30):
    if not frames:
        print(f"[gif] no frames, skipping {path}")
        return
    import imageio.v2 as imageio
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, frames, fps=fps)
    print(f"[gif] wrote {path}  ({len(frames)} frames)")

def plot_curve(path="outputs/mspacman_returns.npy", out="outputs/curve.png"):
    """Plot a saved training-return log (raw and 20-epoch moving average)."""
    r = np.load(path)
    plt.figure()
    plt.plot(r, alpha=0.3, label="return")
    if len(r) >= 20:
        plt.plot(np.convolve(r, np.ones(20) / 20, mode="valid"), label="smoothed")
    plt.xlabel("epoch"); plt.ylabel("return"); plt.legend()
    plt.savefig(out)
    plt.close()
    print(f"[plot] wrote {out}")

def danger_heatmap_frame(danger, player_px, enemy_px, snap, t, enemy_active=None, auto_scale=True, danger_max=DANGER_MAX):
    """Danger heatmap as an RGB array. Positions are pixel coords; `snap` maps them to cells."""
    d = np.asarray(danger)
    if auto_scale:
        vmin, vmax = float(d.min()), float(d.max())
        if vmax - vmin < 1e-6:           # all-zero field (gate off): avoid a degenerate scale
            vmax = vmin + 1.0
    else:
        vmin, vmax = 0.0, danger_max
 
    fig, ax = plt.subplots(figsize=(5, 4), dpi=HEATMAP_DPI)
    im = ax.imshow(d.T, origin="upper", cmap="hot", vmin=vmin, vmax=vmax, interpolation="nearest")       # .T: fields are indexed [x, y]
    plt.colorbar(im, ax=ax, label="danger")
 
    px, py = snap(jnp.asarray(player_px))
    ax.plot(int(px), int(py), "co", markersize=8)
 
    enemy_px = np.asarray(enemy_px)
    active = (np.ones(len(enemy_px), bool) if enemy_active is None
              else np.asarray(enemy_active) > 0)
    for (ex, ey), a in zip(enemy_px, active):
        if a:                            # only mark enemies that currently count as dangerous
            gx, gy = snap(jnp.array([ex, ey]))
            ax.plot(int(gx), int(gy), "b+", markersize=10, markeredgewidth=2)
 
    ax.set_title(f"danger t={t}  [{vmin:.2f}, {vmax:.2f}]")
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return frame

# ----- Main -----
def main(game="mspacman", mode="render", maze_id=0, mazes=None, seed=0, n_seeds=10, danger="net", weights=WEIGHTS, max_steps=MAX_STEPS):
    
    #maze_id validator
    if game == "mspacman":
        if maze_id > 3:
            print(f"[debug] {maze_id} is not within 0-3 for game 'mspacman'. Please select an acceptable maze")
            return 
    
    if mode == "render":
        env = make_env(game, maze_id)
        enc = ENCODERS[game](env, maze_id=maze_id)
        danger_fn = make_danger_fn(danger, env, enc, weights)
        score, frames, heatmap = run(env, enc, danger_fn, seed=seed, max_steps=max_steps)
        tag = f"{game}_{danger}_{maze_id}_{seed}"
        print(f"[render] {game} maze={maze_id} seed={seed} danger={danger}  "
              f"score={score}  steps={len(frames)}")
        save_gif(frames, f"{OUT_DIR}/{tag}.gif")
        save_gif(heatmap, f"{OUT_DIR}/heatmap_{tag}.gif", fps=15)
        return score
 
    mazes = mazes if mazes is not None else range(N_MAZES[game])
    return score_mazes(game, mazes, n_seeds, danger, weights, max_steps)
 
 
if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--game", default="mspacman", choices=list(ENCODERS))
    p.add_argument("--mode", default="render", choices=["render", "score"])
    p.add_argument("--maze", type=int, default=0, help="maze for render mode")
    p.add_argument("--mazes", type=int, nargs="+", default=None, help="mazes for score mode (default: all)")
    p.add_argument("--seed", type=int, default=0, help="seed for render mode")
    p.add_argument("--n-seeds", type=int, default=10, help="seeds per maze in score mode")
    p.add_argument("--danger", default="net", choices=["net", "handcrafted", "none"])
    p.add_argument("--weights", default=WEIGHTS)
    p.add_argument("--max-steps", type=int, default=MAX_STEPS)
    a = p.parse_args()
    main(game=a.game, mode=a.mode, maze_id=a.maze, mazes=a.mazes, seed=a.seed, n_seeds=a.n_seeds, danger=a.danger, weights=a.weights, max_steps=a.max_steps)