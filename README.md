# Treasure Game

The Treasure Game is a video game in which an agent must navigate a dungeon to collect gold. The details of the game were first described in this paper: 


G.D. Konidaris, L.P. Kaelbling, and T. Lozano-Perez. *Symbol Acquisition for Probabilistic High-Level Planning*. In Proceedings of the Twenty Fourth International Joint Conference on Artificial Intelligence, pages 3619-3627, July 2015.


Requirements:
- Python 3
- OpenAI Gym
- NumPy
- PyGame


## Basic Usage

After installing the environment, it can simply be used in the standard gym-like fashion:

```
import gym
import gym_treasure_game

env = gym.make('treasure_game-v0')
for episode in range(5):
    state = env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        env.render('human')
        if done:
            break
```

The environment 

## Wrappers

By default, the state space is given by a vector describing the position of the agent, and the state of the objects in the world. If you would like to use the pixel information as the state space, you can use the `ObservationWrapper` provided like so 

```
import gym
import gym_treasure_game
from gym_treasure_game.envs.treasure_game import ObservationWrapper

env = ObservationWrapper(gym.make('treasure_game-v0'))
obs = env.reset()  # returns the RGB array
```
