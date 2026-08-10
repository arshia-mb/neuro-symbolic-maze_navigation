# Neuro-Symbolic Ms. Pac-Man

A neuro-symbolic agent for maze navigation with enemy avoidance in [JAXAtari](https://github.com/k4ntz/JAXAtari), as part of the Practical AI lab for summer 2026 at TU Darmstadt.

## The idea
Navigation and enemy avoidance are two different problems. Navigation on a known maze is a shortest-path problem solved exactly by planning; no learning needed. Enemy avoidance is where the difficulty lives, so that is the only part we learn.

## Files
- `navigation.py` — symbolic navigator only (no learning, no avoidance). Usable standalone as a pathfinding baseline.
- `agent_REINFORCE_v1.py` — the full agent: learned danger field + planner, trained with REINFORCE through the differentiable planner.
- `utility.py` — utilities used for evaluation, debugging, and visualization.
