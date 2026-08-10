"""
Shared utilities: a generic rollout that runs ANY model, and a GIF writer for visualization and testing of the models. 

The rollout is model-agnostic. You pass an `act_fn` with the signature
    act_fn(obs, carry) -> (action, carry)

Note: this rollout is a plain Python loop that steps the (internally jitted) env and collects rendered frames. It is meant for evaluation / visualisation, NOT for training. It is intentionally not jitted, because rendering and GIF export are host-side operations.
"""
from typing import Any, Callable, Tuple

import numpy as np
import jax


def rollout(env, act_fn: Callable, init_carry: Any, seed: int = 0,
            max_steps: int = 3000, render: bool = True):
    """Run one episode driving the env with `act_fn`.

    env        -- JAXAtari env
    act_fn     -- (obs, carry) -> (action, carry)
    init_carry -- initial per-step state for act_fn (model-specific)
    render     -- if True, collect RGB frames for a GIF

    returns dict with: score, steps, frames (list or None)
    """
    obs, state = env.reset(jax.random.PRNGKey(seed))
    carry = init_carry
    frames = [] if render else None

    for _ in range(max_steps):
        action, carry = act_fn(obs, carry)
        obs, state, reward, done, info = env.step(state, action)

        if render:
            frames.append(np.asarray(env.render(state), dtype=np.uint8))
        if bool(done):
            break

    return {
        "score": int(state.score),
        "steps": len(frames) if frames is not None else None,
        "frames": frames,
    }


def save_gif(frames, path: str, fps: int = 30) -> bool:
    """Write frames to a GIF. Returns True on success, False if imageio missing."""
    try:
        import imageio.v2 as imageio
    except ImportError:
        print("[gif] imageio not installed; run `pip install imageio` to enable GIFs")
        return False
    imageio.mimsave(path, frames, fps=fps)
    print(f"[gif] wrote {path}  ({len(frames)} frames)")
    return True