# Treasure Game

The Treasure Game is a video game in which an agent must navigate a dungeon to collect gold,
and then return to the top of the screen to succeed. The details of the game were first described in this paper: 


G.D. Konidaris, L.P. Kaelbling, and T. Lozano-Perez. *Symbol Acquisition for Probabilistic High-Level Planning*. In Proceedings of the Twenty Fourth International Joint Conference on Artificial Intelligence, pages 3619-3627, July 2015.


Requirements:
- Python 3
- OpenAI Gym
- NumPy
- PyGame

To install the environment, clone this repo and install it with pip:

```
git clone https://github.com/sd-james/gym-treasure-game.git
cd gym-treasure-game
pip install -e .
```

## Basic Usage

After installing the environment, it can simply be used in the standard gym-like fashion:

```python
from gym_treasure_game import make_env  # convenience function

env = make_env('treasure_game-v0')
for episode in range(5):
    state = env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        env.render('human')
        if done:
            break
```

## Wrappers

By default, the state space is given by a vector describing the position of the agent, and the state of the objects in the world. If you would like to use the pixel information as the state space, you can use the `ObservationWrapper` provided like so 

```
from gym_treasure_game import make_env
from gym_treasure_game.envs.treasure_game import ObservationWrapper

env = ObservationWrapper(make_env('treasure_game-v0'))
obs = env.reset()  # returns the RGB array
```
