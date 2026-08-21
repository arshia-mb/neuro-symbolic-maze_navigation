# Neuro-Symbolic Ms. Pac-Man

A neuro-symbolic agent for maze navigation with enemy avoidance in [JAXAtari](https://github.com/k4ntz/JAXAtari), as part of the Practical AI lab for summer 2026 at TU Darmstadt.

## The idea
Navigation and enemy avoidance are two different problems. Navigation on a known maze is a shortest-path problem solved exactly by planning; no learning needed. Enemy avoidance is where the difficulty lives, so that is the only part we learn.

## Files
- `navigation.py` — symbolic navigator only.
- `agent.py` — the full agent + the learnable danger fields. 
- `test.py` — for testing, debugging, and visualization. 

The `*_encoder.py` files are encoder heads for each game the agent is tested or trained on. 
