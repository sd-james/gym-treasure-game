from gym.envs.registration import register

register(
    id='treasure_game-v0',
    entry_point='gym_multi_treasure_game.envs:TreasureGame',
)