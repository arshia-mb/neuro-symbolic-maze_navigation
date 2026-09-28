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


## Run Instructions

Use `test.py` to test and run the model against the already fully setup game environments (Ms. Pacman, Bankheist, Pacman).

Testing has been made very simple with a CLI system to select and customize the run to the users preferences. like so:
```bash
python3 src/test.py --game mspacman --mode render    #default
```

**CLI Arguments**
| Arguments     | defaults          | choices                                                          | help                                                                                                          |
|---------------|-------------------|------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| `--game`      | mspacman          | [s] mspacman,<br> bankheist,<br> pacman                        | The games (encoder) of which to run on                                                                        |
| `--mode`      | render            | [s] render,<br> score                                          | 'render' runs and renders along the output gif and heatmap,<br> 'score' runs and output only the gotten score |
| `--maze`      | 0                 | [int] 0-3 for mspacman,<br> 0-4 for Bankheist,<br> 04 for pacman | maze id selection for game runs                                                                               |
| `--mazes`     | 0                 | [int] IDK                                                        | IDK                                                                                                           |
| `--seed`      | 0                 | [int] 0-6                                                        | specifies the seed for render mode                                                                            |
| `--n-seeds`   | 10                | [int]                                                         | seeds per maze in score mode                                                                                  |
| `--danger`    | net               | [s] net, handcrafted, none                                     | The make danger controller param                                                                              |
| `--weights`   | WEIGHTS(constant) | [path]                                                            | The path to the weights to be used for testing                                                                |
| `--max-steps` | 3000 (300 cycles) | [int]                                                            | The maximum steps for a test run                                                                              |


## Environment note (WSL + CUDA)

If convolutions throw `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`, remove stray CUDA-13 packages and realign JAX's CUDA stack:

```bash
pip uninstall -y nvidia-cudnn-cu13 nvidia-nccl-cu13 nvidia-nvshmem-cu13 nvidia-cusparselt-cu13
pip install --force-reinstall "jax[cuda12]==0.10.0"
```

The `cuda_executor ... Version does not match the format X.Y.Z` warning under WSL is cosmetic.