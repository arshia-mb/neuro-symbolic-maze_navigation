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

from .msp_nav import plan, greedy_action
from .mspacman_encoder import make_mspacman_encoder

from ..utility import save_gif, danger_heatmap_frame

#MAZE_ID = 3
RESET_LEVEL_TO_MAZE = {0: 1, 1: 3, 2: 5, 3: 7} # maze to level const
MAX_STEPS = 3000
SEED = 0
LAMBDA = 8.0   
LARGE_COST = 1e6     



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

def audit_features(enc, state, obs, maze, walkable, snap):
    f = enc.features(state, obs, maze, walkable)
    print("feature shape:", f.shape, "  nan:", bool(jnp.isnan(f).any()), "  inf:", bool(jnp.isinf(f).any()))

    names = ["DOF up", "DOF right", "DOF left", "DOF down", "dist-to-pac", "ghost occ", "ghost dx", "ghost dy"]
    for c in range(f.shape[-1]):
        ch = f[..., c]
        print(f"ch{c} {names[c]:14s}: min {float(ch.min()):8.2f}  max {float(ch.max()):8.2f} mean {float(ch.mean()):6.2f}  nonzero {int((ch != 0).sum())}")

    # cross-checks:
    gpos = obs.ghost_positions
    ggx = (gpos[:, 0] + 5) // 4
    ggy = (gpos[:, 1] + 3) // 4
    print("\n-- ghost cross-check --")
    print("ghost cells:", [(int(ggx[i]), int(ggy[i])) for i in range(gpos.shape[0])])
    occ = f[..., 5]
    print("occ sum:", float(occ.sum()), "(should ≈ #distinct ghost cells)")
    for i in range(gpos.shape[0]):
        gx_, gy_ = int(ggx[i]), int(ggy[i])
        print(f"  ghost {i} at ({gx_},{gy_}): occ={float(occ[gx_,gy_])}  "
              f"dx={float(f[gx_,gy_,6])}  dy={float(f[gx_,gy_,7])}")

# --- Game Test ---
def run(env, enc, danger_fn, lam=LAMBDA, seed=SEED, max_steps=MAX_STEPS, render=True, heatmap_every=5):
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
    info = None

    for t in range(max_steps):
        action, prev_dir, V_nav, danger = decide(obs, state, prev_dir)

        obs, state, reward, done, info = env.step(state, action)

        if render:
            frames.append(np.asarray(env.render(state), dtype=np.uint8))
            if t % heatmap_every == 0:              # matplotlib rendering is slow — sample it
                heatmap.append(danger_heatmap_frame(danger, obs, snap, t))

        if bool(done):
            break

    return int(state.score), frames, heatmap, info



def set_run_maze(maze_id=0):
    #imports
    from jaxatari.games.jax_mspacman import JaxPacman, MsPacmanConstants
    
    MAZE_ID = maze_id   # Maze ID (duh)
    
    #parm init
    consts = MsPacmanConstants(RESET_LEVEL=RESET_LEVEL_TO_MAZE[MAZE_ID])
    
    
    env = JaxPacman(consts=consts)
    enc = make_mspacman_encoder(env, maze_id=MAZE_ID)
    
    # danger source - switch this for the test
    #danger_fn = handcrafted_danger(enc.snap, threat=THREAT)
    danger_fn = make_net_danger_fn(env, enc, "outputs/weights/mspacman_v4.msgpack")
    
    return env, enc, danger_fn
    
# --- Danger fn ---
def make_net_danger_fn(env, enc, model_path):
    from ..agent_MF import DangerNet, load_params 
    net = DangerNet()
    obs, state = env.reset(jax.random.PRNGKey(0))   # you have obs already in main
    template = net.init(jax.random.PRNGKey(0), enc.features(state, obs, enc.maze, enc.walkable))
    params = load_params(template, model_path)
    def danger_fn(obs, state):
        return net.apply(params, enc.features(state, obs, enc.maze, enc.walkable))
    return danger_fn
    
# --- Main ---
def main():
    #test environment and related game encoder head
    #env = jaxatari.make("mspacman")
    
    #imports
    from jaxatari.games.jax_mspacman import get_level_maze
    import time

    t_start = t_run_old = time.time()
    for maze in RESET_LEVEL_TO_MAZE:
        
        print(f"\nlog: === TEST MAZE {maze} ===")       # tags inspired by esther
        env, enc, danger_fn = set_run_maze(maze_id=maze)
    

        score, frames, heatmap, info = run(env, enc, danger_fn, lam=LAMBDA, render=True)

        final_level = int(info.level)
        final_maze = int(get_level_maze(final_level))
        final_lives = int(info.lives)
        maze_changed = final_maze != maze

        print(f"[test] start maze={maze}  lambda={LAMBDA}")
        print(f"[test] score={score}  frames={len(frames)}  lives left={final_lives}")
        print(f"[test] ended on level={final_level}  maze={final_maze}"
            + ("  (maze changed mid-run!)" if maze_changed else ""))

        save_gif(frames, f"outputs/mspacman/msp_maze-{final_maze}_s-{score}_debug.gif")
        save_gif(heatmap, f"outputs/mspacman/msp_maze-{final_maze}_s-{score}_heatmap.gif", fps=15)
        t_run = time.time()
        print(f"log: round time for run maze_id={maze}: {round(t_run - t_run_old, 2)} seconds")
        t_run_old = t_run
    
    t_end = time.time()
    print(f"log: Total runtime: {round(t_end - t_start, 2)} seconds")
    
    
    
    
    
    #plot_curve()
    
 
if __name__ == "__main__":
    main()