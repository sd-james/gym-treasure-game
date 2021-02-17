import gym
from gym.spaces import Discrete

from gym_multi_treasure_game.envs.pca.base_pca import PCA_STATE, PCA_INVENTORY
from s2s.env.s2s_env import MultiViewEnv, View


class MockTreasureGame(MultiViewEnv):

    def __init__(self, version, **kwargs):
        self.option_names = ['go_left_option', 'go_right_option', 'up_ladder_option', 'down_ladder_option',
                             'interact_option', 'down_left_option', 'down_right_option', 'jump_left_option',
                             'jump_right_option']
        self.action_space = Discrete(len(self.option_names))
        self.name = "TreasureGameV{}".format(version)
        self.version = version

    def __str__(self):
        return self.name

    def n_dims(self, view: View, flat=False) -> int:
        """
        The dimensionality of the state space, depending on the view
        """
        if view == View.PROBLEM:
            if self.version == 10:
                return 2
            return 3
        if flat:
            return PCA_STATE + PCA_INVENTORY
        return 2

    def describe_option(self, option: int) -> str:
        return self.option_names[option]

    @property
    def agent_space(self) -> gym.Space:
        raise NotImplementedError

    @property
    def object_space(self) -> gym.Space:
        raise NotImplementedError

    def current_agent_observation(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def render(self, mode='human'):
        raise NotImplementedError
