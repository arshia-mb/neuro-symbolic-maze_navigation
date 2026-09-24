import jax
import jax.numpy as jnp
from jax import lax
from numpy.random import seed

from encoders.game_encoder import GameEncoder
import jaxatari.games.jax_alien as game
import alien_encoder as alien_encoder
from alien_encoder import make_alien_encoder



TILE = 4
LARGE_COST = 1e6
PLAYER_FOOTPRINT = (8, 13)


env = jaxatari.make("alien")
obs, state = env.reset(jax.random.PRNGKey(seed))
encoder = make_alien_encoder(env)

eggs = game.state.eggs

gx, gy = alien_encoder._egg_indices(eggs, encoder.walkable.shape)

print("walkable shape:", encoder.walkable.shape)

print("first 10 egg positions:")
print(eggs[0, :10, :2])

print("first 10 egg grid positions:")
print(gx[0, :10])
print(gy[0, :10])

print("walkable at egg positions:")
print(encoder.walkable[gx[0, :10], gy[0, :10]])