import jax, jaxatari
from encoders.pacman_encoder import make_pacman_encoder, pellet_indices, pos_to_grid

env = jaxatari.make("pacman")
enc = make_pacman_encoder(env, maze_id=0)
obs, state = env.reset(jax.random.PRNGKey(0))

print("pellets shape:", obs.pellets.shape)          # expect (18, 8)
print("maze shape:", enc.maze.shape)                 # is it (40,44,4)?
print("pellets true:", int(obs.pellets.sum()))
goals = enc.goal(obs, enc.walkable, enc.gx, enc.gy)
print("goal mask sum:", int(goals.sum()))            # should ≈ pellets true
print("player cell:", pos_to_grid(obs.player_position), "walkable there:", 
      bool(enc.walkable[pos_to_grid(obs.player_position)]))

import numpy as np

# print a handful of raw pellet coords and compare to where the player/walkable cells are
gx, gy = enc.gx, enc.gy   # the precomputed pellet index arrays, shape (18, 8) each
px_idx, py_idx = np.where(np.asarray(obs.pellets))   # which (row,col) IN THE PELLET GRID actually have pellets
print("sample pellet grid cells with pellets:", list(zip(px_idx[:5], py_idx[:5])))
print("their mapped maze cells:", [(int(gx[px_idx[i], py_idx[i]]), int(gy[px_idx[i], py_idx[i]])) for i in range(5)])
print("walkable at those maze cells:", [bool(enc.walkable[int(gx[px_idx[i],py_idx[i]]), int(gy[px_idx[i],py_idx[i]])]) for i in range(5)])