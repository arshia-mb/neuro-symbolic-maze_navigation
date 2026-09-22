"""Smoke test for the Alien encoder and shared symbolic navigation."""

import jax
import jax.numpy as jnp
import jaxatari

from encoders.alien_encoder import make_alien_encoder
from navigation import greedy_action, plan


MAX_STEPS = 100
SEED = 0
LAMBDA = 8.0


def run(max_steps=MAX_STEPS, seed=SEED):
    env = jaxatari.make("alien")
    obs, state = env.reset(jax.random.PRNGKey(seed))
    encoder = make_alien_encoder(env)

    assert encoder.maze.shape == encoder.walkable.shape + (4,)
    features = encoder.features(
        state, obs, encoder.maze, encoder.walkable
    )
    assert features.shape == encoder.walkable.shape + (8,)
    assert bool(jnp.isfinite(features).all())

    previous_direction = jnp.int32(0)
    for step in range(max_steps):
        goals = encoder.goal(
            obs,
            encoder.walkable,
            encoder.gx,
            encoder.gy,
            state,
        )
        assert goals.shape == encoder.walkable.shape

        if int(goals.sum()) == 0:
            break

        value = plan(encoder.maze, goals, encoder.walkable)
        danger = jnp.zeros_like(value)
        position = jnp.array([obs.player.x, obs.player.y])
        action, previous_direction = greedy_action(
            value,
            danger,
            encoder.maze,
            goals,
            position,
            previous_direction,
            encoder.snap,
            LAMBDA,
        )

        assert 2 <= int(action) <= 5
        obs, state, _, done, _ = env.step(state, action)
        if bool(done):
            break
    else:
        step = max_steps - 1

    return encoder, obs, state, step + 1


def main():
    encoder, obs, state, steps = run()
    features_shape = encoder.features(
        state,
        obs,
        encoder.maze,
        encoder.walkable,
    ).shape
    print("walkable shape:", encoder.walkable.shape)
    print("maze shape:", encoder.maze.shape)
    print("features shape:", features_shape)
    print("steps:", steps)
    print("final score:", int(jnp.ravel(state.level.score)[0]))
    print("remaining eggs:", int(jnp.sum(state.eggs[:, :, 2])))


if __name__ == "__main__":
    main()