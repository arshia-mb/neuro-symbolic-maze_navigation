import jax, jax.numpy as jnp, jaxatari, numpy as np
from navigation import plan, greedy_action
from encoders.bankhiest_encoder import make_bankheist_encoder

env = jaxatari.make("bankheist")
obs, state = env.reset(jax.random.PRNGKey(0))
enc = make_bankheist_encoder(state)
initial_map = state.map_collision

frac = 100 * float(enc.walkable.sum()) / enc.walkable.size
print(f"walkable {enc.walkable.shape}  maze {enc.maze.shape}")
print(f"walkable cells: {int(enc.walkable.sum())} / {enc.walkable.size}  ({frac:.1f}%)")

zero_danger = lambda obs, state: jnp.zeros(enc.walkable.shape)

prev_dir = jnp.int32(0)
frames = []
prev_goals_n = None
prev_lives = int(state.player_lives)
map_changed_at = None

for t in range(600):
    goals = enc.goal(obs, enc.walkable, enc.gx, enc.gy)
    n_goals = int(goals.sum())
    V = plan(enc.maze, goals, enc.walkable)
    dng = zero_danger(obs, state)
    pos = jnp.array([obs.player.x, obs.player.y])
    action, prev_dir = greedy_action(V, dng, enc.maze, goals, pos, prev_dir, enc.snap, lam=0.0)

    if map_changed_at is None and not jnp.array_equal(state.map_collision, initial_map):
        map_changed_at = t
        print(f"*** map_collision changed at t={t} -- encoder is now stale (level transition?) ***")

    if prev_goals_n is not None and n_goals != prev_goals_n:
        print(f"*** active goal (bank) count changed {prev_goals_n} -> {n_goals} at t={t} ***")
    prev_goals_n = n_goals

    lives_now = int(state.player_lives)
    if lives_now != prev_lives:
        print(f"*** player_lives changed {prev_lives} -> {lives_now} at t={t} (death/respawn) ***")
    prev_lives = lives_now

    if t % 25 == 0 or n_goals == 0:
        px, py = enc.snap(pos)
        nav4 = [float(V[px, py-1]), float(V[px+1, py]), float(V[px-1, py]), float(V[px, py+1])]
        print(f"t={t:3d} cell=({int(px):2d},{int(py):2d}) V_here={float(V[px,py]):.0f} "
              f"nav(u,r,l,d)={[round(n,0) for n in nav4]} action={int(action)} "
              f"money={int(state.money)} goals={n_goals} lives={lives_now}")

    obs, state, r, done, info = env.step(state, action)
    frames.append(np.asarray(env.render(state), dtype=np.uint8))
    if bool(done):
        print(f"done at t={t}")
        break

import imageio.v2 as imageio
imageio.mimsave("gifs/bh_nav_only_v2.gif", frames, fps=30)
print(f"final money={int(state.money)}  frames={len(frames)}")