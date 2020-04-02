import enum
from abc import abstractmethod
from typing import Any, Dict, Tuple

import gym


class View(enum.Enum):
    PROBLEM = 1,
    AGENT = 2,


class MultiViewEnv(gym.Env):

    @abstractmethod
    def n_dims(self, view: View) -> int:
        """
        The dimensionality of the state space, depending on the view
        """
        pass

    @property
    @abstractmethod
    def agent_space(self) -> gym.Space:
        """
        The agent space size
        """
        pass

    @abstractmethod
    def current_agent_observation(self):
        pass

    def reset(self) -> Tuple[Any, Any]:
        state = super().reset()
        observation = self.current_agent_observation()
        return state, observation

    def step(self, action) -> Tuple[Any, Any, float, bool, Dict]:
        """
        Take a step in the environment
        :param action: the action to execute
        :return: the state, agent's observation, reward, done flag and info
        """
        state, reward, done, info = super().step(action)
        observation = self.current_agent_observation()
        return state, observation, reward, done, info
