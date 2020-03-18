# Treasure Game

The Treasure Game is a video game in which an agent must navigate a dungeon to collect gold,
and then return to the top of the screen to succeed. The details of the game were first described in this paper: 

G.D. Konidaris, L.P. Kaelbling, and T. Lozano-Perez. *Symbol Acquisition for Probabilistic High-Level Planning*. In Proceedings of the Twenty Fourth International Joint Conference on Artificial Intelligence, pages 3619-3627, July 2015.

To ensure that the dynamics remain precisely that of the original paper, I have not modified the core code in any way. 
Changes that were made include upgrading from Python2 to Python3, and redoing the sprites (which are used with permission 
from Jonathan Blow).


## Installing

Requirements:
- Python 3 (3.7)
- OpenAI Gym (0.15.6)
- NumPy (1.18.1)
- PyGame (1.9.6)

The combination of package versions above is guaranteed to work, but there should be no issue with using any new version
of the above.

To install the environment, clone this repo and install it with pip:

```
git clone https://github.com/sd-james/gym-treasure-game.git
cd gym-treasure-game
pip install -e .
```

## Basic Usage

After installing the environment, it can simply be used in the standard gym-like fashion:

```python
import gym

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

## Wrappers

By default, the state space is given by a vector describing the position of the agent, and the state of the objects in the world. If you would like to use the pixel information as the state space, you can use the `ObservationWrapper` provided like so 

```python
import gym
from gym_treasure_game.envs.treasure_game import ObservationWrapper

env = ObservationWrapper(gym.make('treasure_game-v0'))
obs = env.reset()  # returns the RGB array
```

## Acknowledgements

With thanks to [George Konidaris](http://cs.brown.edu/people/gdk) and [Cam Allen](http://camallen.net/).