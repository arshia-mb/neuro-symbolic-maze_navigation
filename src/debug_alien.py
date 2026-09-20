import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jaxatari.games.jax_alien as game
import encoders.alien_encoder as enc


frame = np.asarray(game.env.render(game.state))
coll = np.asarray(game.state.map_collision)      # <-- replace with Alien's collision field
wk = np.asarray(enc.walkable)

fig, ax = plt.subplots(1, 3, figsize=(15, 6))
ax[0].imshow(frame);                                ax[0].set_title("game frame")
ax[1].imshow(coll.T, cmap="gray");                  ax[1].set_title(f"collision map {coll.shape}")
ax[2].imshow(wk.T, cmap="gray", interpolation="nearest"); ax[2].set_title(f"walkable {wk.shape}")
plt.tight_layout()
plt.savefig("alien_walkable_check.png", dpi=110)
