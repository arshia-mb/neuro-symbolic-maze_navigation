# Neuro-Symbolic Ms. Pac-Man

A neuro-symbolic agent for maze navigation with enemy avoidance in [JAXAtari](https://github.com/k4ntz/JAXAtari), as part of the Practical AI lab for summer 2026 at TU Darmstadt.

## The idea

Navigation and enemy avoidance are two different problems. **Navigation** on a known maze is a shortest-path problem — solved exactly by planning, no learning needed. 
**Enemy Avoidance** is where the difficulty lives, so that is the only part we learn. The agent runs two separate fields and combines them at decision time:

The agent picks the move that minimises `nav_cost + λ · danger`. `λ` is the safety-vs-pellets knob. 

## Files

Source files:
- `navigation.py` — the shared symbolic core: `plan` (distance-to-goal value iteration) and `greedy_action` (the `nav + λ·danger` combine).
- `agent_MF.py` — the model-free agent: the `DangerNet` conv net + the REINFORCE training loop that learns the danger field.
- `test.py` — evaluation harness. Loads a trained danger net (or a handcrafted danger field), plays one episode, prints the score, and writes a GIF. The danger source is pluggable via a `danger_fn`.

For each game to be tested or trained on the net an encoder head is needed:
- `mspacman_encoder.py` — the Ms. Pac-Man adapter: pulls features out of the observation and builds the `GameEncoder` the core consumes.


`legacy/` holds earlier single-field versions kept for reference. `examples/gifs/` holds demo GIFs (navigation, danger avoidance, the reward-hack failure, etc.).  `outputs/` holds trained weights and training curves.

## Environment note (WSL + CUDA)

If convolutions throw `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`, remove stray CUDA-13 packages and realign JAX's CUDA stack:

```bash
pip uninstall -y nvidia-cudnn-cu13 nvidia-nccl-cu13 nvidia-nvshmem-cu13 nvidia-cusparselt-cu13
pip install --force-reinstall "jax[cuda12]==0.10.0"
```

The `cuda_executor ... Version does not match the format X.Y.Z` warning under WSL is cosmetic.