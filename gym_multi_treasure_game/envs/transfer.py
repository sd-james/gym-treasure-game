import random

import numpy as np

from gym_multi_treasure_game.envs.evaluate import evaluate_model
from gym_multi_treasure_game.envs.multi_treasure_game import MultiTreasureGame
from gym_multi_treasure_game.envs.pca.base_pca import PCA_STATE, PCA_INVENTORY
from gym_multi_treasure_game.envs.pca.pca import PCA
from gym_multi_treasure_game.envs.pca.pca_wrapper import PCAWrapper
from s2s.core.build_model import build_model
from s2s.env.s2s_env import View
from s2s.utils import Recorder, save, now



def _build_model(env, n_episodes, previous_predicates, previous_operators):
    domain, problem, info = build_model(env,
                                        save_dir=None,
                                        n_jobs=16,
                                        seed=0,
                                        n_episodes=n_episodes,
                                        options_per_episode=1000,
                                        view=View.AGENT,
                                        linking_threshold=0.1,
                                        specify_rewards=True,
                                        effect_epsilon=4,
                                        init_epsilon=4,
                                        generate_positive_samples=True,
                                        low_threshold=0.45,
                                        high_threshold=0.9,
                                        augment_negative=True,
                                        max_precondition_samples=5000,
                                        precondition_c_range=np.logspace(0.01, 0.5, 10),
                                        precondition_gamma_range=np.logspace(0.1, 1, 10),
                                        visualise=True,
                                        verbose=True)


if __name__ == '__main__':

    np.random.seed(0)
    random.seed(0)
    recorder = Recorder()
    transfer_recorder = Recorder()
    max_episodes = 60
    n_experiments = 10
    threshold = 1

    pca = PCA(PCA_STATE)
    pca.load('pca/models/pca_state.dat')
    pca2 = PCA(PCA_INVENTORY)
    pca2.load('pca/models/pca_inventory.dat')

    for experiment in range(n_experiments):

        tasks = list(range(1, 11))
        random.shuffle(tasks)
        previous_operators = list()
        previous_predicates = list()
        for task in tasks:

            for n_episodes in range(max_episodes):
                env = PCAWrapper(MultiTreasureGame(task, pcas=[pca, pca2], split_inventory=True), pca, pca2=pca2)
                linked_pddl, pddl_problem, learned_operators, n_predicates, n_operators = \
                    _build_model(env, n_episodes, previous_predicates, previous_operators)
                score = evaluate_model(task, env, linked_pddl, pddl_problem)
                recorder.record(experiment, task, n_episodes, score)
                print("Experiment: {}\nTask: {}\nSamples: {}\nScore: {}\n"
                      .format(experiment, task, n_episodes, score))
                if score >= threshold:
                    transfer_recorder.record(experiment, task, n_episodes,
                                             (n_predicates, n_operators, len(linked_pddl.propositions),
                                              len(linked_pddl.operators),
                                              len(previous_operators), len(learned_operators)))
                    previous_operators, previous_predicates = transfer(previous_operators, previous_predicates,
                                                                       learned_operators,
                                                                       linked_pddl)
                    save((recorder, transfer_recorder), 'recorders_{}.pkl'.format(now()))
                    if score >= threshold:
                        break
            save((recorder, transfer_recorder), 'recorders_{}.pkl'.format(now()))
