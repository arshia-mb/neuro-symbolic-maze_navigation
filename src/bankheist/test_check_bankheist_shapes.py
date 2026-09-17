#standalone check for Bankheist

import jax
import jax.numpy as jnp
import jaxatari

env = jaxatari.make("mspacman")
key = jax.random.PRNGKey(0)

print("action_space test")
print(env.action_space())   # expect Discrete(9) -claude

print("\nreset() test")
obs, state = env.reset(key)
for field in obs.__dataclass_fields__:
    val = getattr(obs, field)
    print(f"{field:20s} shape={getattr(val, 'shape', None)} dtype={getattr(val, 'dtype', type(val))}")
    if getattr(val, 'shape', None) is not None and val.size <= 20:
        print(f"{'':20s} value={val}")

print("\na few random-action steps")
for t in range(5):
    key, akey = jax.random.split(key)
    action = jax.random.randint(akey, (), 0, env.action_space().n)
    obs, state, reward, done, info = env.step(state, action)
    print(f"step {t}: action={int(action)} reward={float(reward):.2f} done={bool(done)} "
          f"player_position={obs.player_position} ghost_positions[0]={obs.ghost_positions[0]}")

print("\n pellets grid check")
print("pellets shape:", obs.pellets.shape, "dtype:", obs.pellets.dtype)
print("pellets remaining:", int(jnp.sum(obs.pellets)))
print("power_pellets:", obs.power_pellets)

# ahh so thats info
print("\ninfo?")
print(info)
