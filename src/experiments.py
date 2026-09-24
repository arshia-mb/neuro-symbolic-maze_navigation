import jax, jax.numpy as jnp, jaxatari
from navigation import plan
from encoders.bankhiest_encoder import make_bankheist_encoder
from agent_MF import DangerNet, load_params

STEPS = 400
SEED = 0


def run(lam, sticky=1e-3, zero_channels_67=False, normalize_danger=False, steps=STEPS, seed=SEED):
    env = jaxatari.make("bankheist")
    obs, state = env.reset(jax.random.PRNGKey(seed))
    enc = make_bankheist_encoder(state)
    net = DangerNet()
    feats0 = enc.features(state, obs, enc.maze, enc.walkable)
    template = net.init(jax.random.PRNGKey(0), feats0)
    params = load_params(template, "outputs/weights/mspacman_v4.msgpack")

    def danger_fn(obs, state):
        feats = enc.features(state, obs, enc.maze, enc.walkable)
        if zero_channels_67:
            feats = feats.at[..., 6:8].set(0.0)
        return net.apply(params, feats)

    prev_dir = jnp.int32(0)
    visited = set()
    max_money = 0

    for t in range(steps):
        goals = enc.goal(obs, enc.walkable, enc.gx, enc.gy)
        V = plan(enc.maze, goals, enc.walkable)
        danger = danger_fn(obs, state)
        
        # Apply per-frame normalization to the danger field if enabled
        if normalize_danger:
            d_mean = jnp.mean(danger)
            d_std = jnp.std(danger)
            danger = (danger - d_mean) / (d_std + 1e-5)

        pos = jnp.array([obs.player.x, obs.player.y])
        gx, gy = enc.snap(pos)
        visited.add((int(gx), int(gy)))

        nav4 = jnp.array([V[gx, gy - 1], V[gx + 1, gy], V[gx - 1, gy], V[gx, gy + 1]])
        dng4 = jnp.array([danger[gx, gy - 1], danger[gx + 1, gy], danger[gx - 1, gy], danger[gx, gy + 1]])
        score4 = nav4 + lam * dng4
        legal4 = enc.maze[gx, gy]
        score4 = jnp.where(legal4, score4, 1e6)
        tie = jnp.zeros(4).at[prev_dir].add(-sticky)
        d = jnp.argmin(score4 + tie).astype(jnp.int32)
        d = jnp.where(goals[gx, gy], prev_dir, d).astype(jnp.int32)
        action = d + 2
        prev_dir = d

        obs, state, r, done, info = env.step(state, action)
        max_money = max(max_money, int(state.money))
        if bool(done):
            break

    return len(visited), max_money


print(f"{'run_type':<15} {'lambda':>8} {'sticky':>8} {'zero_6_7':>9} {'norm':>6} {'cells_visited':>14} {'max_money':>10}")

print("\n--- coarse lambda sweep (default sticky, channels intact) ---")
for lam in [8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.1, 0.0]:
    n_visited, max_money = run(lam=lam)
    print(f"{'coarse':<15} {lam:8.2f} {1e-3:8.4f} {'no':>9} {'no':>6} {n_visited:14d} {max_money:10d}")

print("\n--- finer lambda scan (1.1 - 1.9) ---")
for lam in [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]:
    n_visited, max_money = run(lam=lam)
    print(f"{'fine':<15} {lam:8.2f} {1e-3:8.4f} {'no':>9} {'no':>6} {n_visited:14d} {max_money:10d}")

"""
print("\n--- per-frame normalized danger ---")
# Testing a wider range since the scale of the danger field is now ~[-3, 3] rather than raw magnitudes
for lam in [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]:
    n_visited, max_money = run(lam=lam, normalize_danger=True)
    print(f"{'normalized':<15} {lam:8.2f} {1e-3:8.4f} {'no':>9} {'yes':>6} {n_visited:14d} {max_money:10d}")

print("\n--- stickier momentum, original lambda ---")
for sticky in [1e-2, 1e-1, 1.0]:
    n_visited, max_money = run(lam=8.0, sticky=sticky)
    print(f"{'sticky_test':<15} {8.0:8.2f} {sticky:8.4f} {'no':>9} {'no':>6} {n_visited:14d} {max_money:10d}")

print("\n--- channels 6-7 zeroed, original lambda ---")
n_visited, max_money = run(lam=8.0, zero_channels_67=True)
print(f"{'zeroed_ch':<15} {8.0:8.2f} {1e-3:8.4f} {'yes':>9} {'no':>6} {n_visited:14d} {max_money:10d}")
"""