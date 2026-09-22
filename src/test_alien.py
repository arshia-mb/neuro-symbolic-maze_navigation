"""Smoke test for the Alien encoder and shared symbolic navigation."""

import argparse
import os

import jax
import jax.numpy as jnp
import jaxatari
import numpy as np

from encoders.alien_encoder import make_alien_encoder
from navigation import greedy_action, plan


MAX_STEPS = 100
SEED = 0
LAMBDA = 8.0


def run(max_steps=MAX_STEPS, seed=SEED, render=False):
    env = jaxatari.make("alien")
    obs, state = env.reset(jax.random.PRNGKey(seed))
    encoder = make_alien_encoder(env)
    frames = [np.asarray(env.render(state), dtype=np.uint8)] if render else None

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
        if render:
            frames.append(np.asarray(env.render(state), dtype=np.uint8))
        if bool(done):
            break
    else:
        step = max_steps - 1

    return encoder, obs, state, step + 1, frames


def save_gif(frames, path, fps=30):
    try:
        import imageio.v2 as imageio
    except ImportError:
        raise RuntimeError(
            "GIF output requires imageio; install it with `pip install imageio`."
        )

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps)


def main(render=False, output="gifs/alien_encoder.gif", max_steps=MAX_STEPS):
    encoder, obs, state, steps, frames = run(
        max_steps=max_steps,
        render=render,
    )
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
    if render:
        save_gif(frames, output)
        print("GIF:", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--render",
        action="store_true",
        help="capture the episode and write a GIF",
    )
    parser.add_argument(
        "--output",
        default="gifs/alien_encoder.gif",
        help="GIF output path used with --render",
    )
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    args = parser.parse_args()
    main(
        render=args.render,
        output=args.output,
        max_steps=args.max_steps,
    )