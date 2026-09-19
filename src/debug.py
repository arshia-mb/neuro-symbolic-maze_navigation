import jax, jax.numpy as jnp, jaxatari, numpy as np
from navigation import plan, greedy_action
from encoders.bankhiest_encoder import make_bankheist_encoder

env = jaxatari.make("bankheist")
obs, state = env.reset(jax.random.PRNGKey(0))
enc = make_bankheist_encoder(state)
initial_map = state.map_collision

print(f"walkable {enc.walkable.shape}  maze {enc.maze.shape}")
print(f"walkable cells: {int(enc.walkable.sum())} / {enc.walkable.size}")

obs, state = env.reset(jax.random.PRNGKey(0))
enc = make_bankheist_encoder(state)
zero_danger = lambda obs, state: jnp.zeros(enc.walkable.shape)

prev_dir = jnp.int32(0)
frames = []
for t in range(600):
    goals = enc.goal(obs, enc.walkable, enc.gx, enc.gy)
    V = plan(enc.maze, goals, enc.walkable)
    dng = zero_danger(obs, state)
    pos = jnp.array([obs.player.x, obs.player.y])
    action, prev_dir = greedy_action(V, dng, enc.maze, goals, pos, prev_dir, enc.snap, lam=0.0)

    if t % 25 == 0:
        px, py = enc.snap(pos)
        nav4 = [float(V[px, py-1]), float(V[px+1, py]), float(V[px-1, py]), float(V[px, py+1])]
        print(f"t={t:3d} cell=({int(px):2d},{int(py):2d}) V_here={float(V[px,py]):.0f} "
              f"nav(u,r,l,d)={[round(n,0) for n in nav4]} action={int(action)} money={int(state.money)}")

    obs, state, r, done, info = env.step(state, action)
    frames.append(np.asarray(env.render(state), dtype=np.uint8))
    if bool(done): break

import imageio.v2 as imageio
imageio.mimsave("gifs/bh_nav_only.gif", frames, fps=30)
print(f"money={int(state.money)}  frames={len(frames)}")