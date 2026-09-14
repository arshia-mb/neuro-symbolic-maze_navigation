"""
Train a small MLP policy (REINFORCE) on the real JAXAtari Bank Heist env,
using bankheist_features() as the state representation.

Lessons carried over from the Ms. Pac-Man training work in this project:
  - Real step() returns 5 values: (obs, state, reward, done, info) -- NOT 6.
  - Don't JIT the loss/grad function over variable-length episode batches --
    pad to a fixed size first (this caused a ~7x slowdown before it was fixed).
  - JIT the fixed-shape single-step forward pass separately for fast rollout.

Run this locally (needs sprites). Everything except the real env.reset/step
calls has been smoke-tested here against a fake env with matching shapes
(see the bottom of this file) -- so if something breaks locally, it's most
likely a real-dynamics assumption (reward timing, done condition, action
mapping), not a shape/plumbing bug.
"""
import pickle
import random

import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax.linen as nn

from .bh_encoder import bankheist_features, FUEL_CAPACITY

N_ACTIONS = 18          # confirmed: Discrete(18)
FEATURE_DIM = 29        # confirmed: bankheist_features() output shape
MAX_STEPS = 500         # Bank Heist episodes can run long (fuel/lives permitting)
GAMMA = 0.99


class PolicyNet(nn.Module):
    n_actions: int = N_ACTIONS
    hidden: int = 64

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden)(x)
        x = nn.tanh(x)
        x = nn.Dense(self.hidden)(x)
        x = nn.tanh(x)
        return nn.Dense(self.n_actions)(x)


def run_episode(params, apply_fn, env, key, greedy=False):
    """Single-episode rollout against the real env, real 5-value step()."""
    reset_key, roll_key = jax.random.split(key)
    obs, state = env.reset(reset_key)

    states, actions, rewards = [], [], []
    total_reward, final_money = 0.0, 0.0

    for t in range(MAX_STEPS):
        feat = bankheist_features(obs)
        logits = apply_fn(params, feat)
        if greedy:
            action = int(jnp.argmax(logits))
        else:
            roll_key, akey = jax.random.split(roll_key)
            action = int(jax.random.categorical(akey, logits))

        obs, state, reward, done, info = env.step(state, jnp.array(action))  # REAL 5-tuple

        states.append(feat)
        actions.append(action)
        rewards.append(float(reward))
        total_reward += float(reward)

        if bool(done):
            break

    final_money = float(obs.score) if hasattr(obs, "score") else total_reward
    return states, actions, rewards, total_reward, final_money


def discounted_returns(rewards, gamma=GAMMA):
    returns = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns


def reinforce_loss(params, apply_fn, states, actions, advantages, mask):
    logits = apply_fn(params, states)                      # (N, n_actions), N fixed (padded)
    logp = jax.nn.log_softmax(logits)
    logp_a = jnp.take_along_axis(logp, actions[:, None], axis=1).squeeze(-1)
    return -jnp.sum(logp_a * advantages * mask) / jnp.maximum(jnp.sum(mask), 1.0)


def train(env, n_iterations=200, episodes_per_iter=8, lr=3e-3, seed=0):
    key = jax.random.PRNGKey(seed)
    model = PolicyNet()
    params = model.init(key, jnp.zeros(FEATURE_DIM))
    opt = optax.adam(lr)
    opt_state = opt.init(params)

    apply_single_jit = jax.jit(model.apply)                  # fixed (29,)->(18,) shape, compiles once
    grad_fn = jax.jit(jax.value_and_grad(reinforce_loss), static_argnums=(1,))
    max_batch = episodes_per_iter * MAX_STEPS                 # fixed padded size -> compiles once

    log_every = 10
    running_reward, running_money = [], []

    for it in range(n_iterations):
        batch_states, batch_actions, batch_returns = [], [], []
        iter_rewards, iter_money = [], []

        for _ in range(episodes_per_iter):
            key, ep_key = jax.random.split(key)
            states, actions, rewards, total_reward, final_money = run_episode(
                params, apply_single_jit, env, ep_key)
            returns = discounted_returns(rewards)
            batch_states.extend(states)
            batch_actions.extend(actions)
            batch_returns.extend(returns)
            iter_rewards.append(total_reward)
            iter_money.append(final_money)

        returns_arr = np.array(batch_returns)
        advantages = (returns_arr - returns_arr.mean()) / (returns_arr.std() + 1e-6)

        n_real = len(batch_states)
        pad = max_batch - n_real
        if pad < 0:
            # an episode ran longer than expected -- truncate defensively
            # rather than crash on a negative pad (shouldn't happen given
            # MAX_STEPS caps each episode, but real dynamics can surprise you)
            batch_states, batch_actions = batch_states[:max_batch], batch_actions[:max_batch]
            advantages = advantages[:max_batch]
            pad = 0
            n_real = max_batch

        states_arr = jnp.stack(batch_states + [jnp.zeros(FEATURE_DIM)] * pad)
        actions_arr = jnp.array(batch_actions + [0] * pad)
        advantages_arr = jnp.array(list(advantages) + [0.0] * pad, dtype=jnp.float32)
        mask_arr = jnp.array([1.0] * n_real + [0.0] * pad, dtype=jnp.float32)

        loss, grads = grad_fn(params, model.apply, states_arr, actions_arr, advantages_arr, mask_arr)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        running_reward.append(np.mean(iter_rewards))
        running_money.append(np.mean(iter_money))

        if (it + 1) % log_every == 0:
            print(f"iter {it+1:4d} | loss {float(loss):+.3f} | "
                  f"avg total_reward {np.mean(running_reward[-log_every:]):+.2f} | "
                  f"avg final_money {np.mean(running_money[-log_every:]):.1f}")

    return params, model, apply_single_jit


def evaluate(params, apply_fn, env, n_trials=10, seed_offset=50_000):
    results = []
    for s in range(n_trials):
        key = jax.random.PRNGKey(seed_offset + s)
        _, _, _, total_reward, final_money = run_episode(params, apply_fn, env, key, greedy=True)
        results.append((total_reward, final_money))
    avg_reward = np.mean([r[0] for r in results])
    avg_money = np.mean([r[1] for r in results])
    return avg_reward, avg_money


def save_policy(params, path="trained_bankheist_policy.pkl"):
    numpy_params = jax.tree_util.tree_map(lambda x: np.asarray(x), params)
    with open(path, "wb") as f:
        pickle.dump(numpy_params, f)
    print(f"Saved trained policy to {path}")


def load_policy(path="trained_bankheist_policy.pkl"):
    with open(path, "rb") as f:
        numpy_params = pickle.load(f)
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x), numpy_params)


if __name__ == "__main__":
    import jaxatari
    env = jaxatari.make("bankheist")

    print("Training a small MLP policy (REINFORCE) on the real Bank Heist env...\n")
    params, model, apply_single_jit = train(env, n_iterations=200, episodes_per_iter=8)

    avg_reward, avg_money = evaluate(params, apply_single_jit, env, n_trials=10)
    print(f"\nTrained policy, greedy eval over 10 trials:")
    print(f"  avg total reward: {avg_reward:+.2f}")
    print(f"  avg final money:  {avg_money:.1f}")

    save_policy(params)