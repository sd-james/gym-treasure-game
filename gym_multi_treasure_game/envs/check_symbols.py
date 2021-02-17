import traceback
from collections import defaultdict

from tqdm import trange, tqdm

from gym_multi_treasure_game.envs.configs import CONFIG
from gym_multi_treasure_game.envs.evaluate import evaluate_manually, evaluate_with_network, validate, \
    evaluate_similarity
from gym_multi_treasure_game.envs.mock_env import MockTreasureGame
from gym_multi_treasure_game.envs.multi_treasure_game import MultiTreasureGame
from gym_multi_treasure_game.envs.pca.base_pca import PCA_STATE, PCA_INVENTORY
from gym_multi_treasure_game.envs.pca.pca import PCA
from gym_multi_treasure_game.envs.pca.pca_wrapper import PCAWrapper
from pyddl.hddl.hddl_domain import HDDLDomain
from pyddl.pddl.domain import Domain
from pyddl.pddl.predicate import Predicate
from s2s.env.s2s_env import View
from s2s.hierarchy.discover_hddl_methods import discover_hddl_tasks
from s2s.planner.mgpt_planner import mGPT
from s2s.portable.build_model_transfer import build_transfer_model
from s2s.portable.problem_symbols import _ProblemProposition
from s2s.portable.transfer import extract
from s2s.render import visualise_symbols
from s2s.utils import make_path, save, load, Recorder, now, range_without
import numpy as np


def try_build(save_dir, task, n_episodes, previous_predicates, previous_operators, verbose=False):
    env = MockTreasureGame(task)

    # pca = PCA(PCA_STATE)
    # pca.load('pca/models/dropped_key_pca_state.dat')
    # pca2 = PCA(PCA_INVENTORY)
    # pca2.load('pca/models/dropped_key_pca_inventory.dat')
    # env = PCAWrapper(MultiTreasureGame(task, pcas=[pca, pca2], split_inventory=True), pca, pca2=pca2)

    if len(previous_predicates) == 0:
        previous_predicates = None
    if len(previous_operators) == 0:
        previous_operators = None

    domain, problem, info = build_transfer_model(env, previous_predicates, previous_operators,
                                                 reload=True,
                                                 save_dir=save_dir,
                                                 n_jobs=16,
                                                 seed=None,
                                                 n_episodes=n_episodes,
                                                 options_per_episode=1000,
                                                 view=View.AGENT,
                                                 **CONFIG[task],
                                                 visualise=False,
                                                 save_data=False,
                                                 verbose=verbose)
    return env, domain, problem, info['n_samples'], info['copied_symbols'], info['copied_operators']


def get_transferable_symbols(domain: Domain, previous_predicates, previous_operators):
    curr_preds, curr_ops = extract(domain)
    predicates = set(previous_predicates)
    predicates.update(curr_preds)
    operators = set(previous_operators)
    operators.update(curr_ops)
    return list(predicates), list(operators)


if __name__ == '__main__':

    dir = '/media/hdd/full_treasure_data/treasure_data'

    # tasks = range_without(1, 11)
    # previous_predicates = list()
    # previous_operators = list()
    # preds = defaultdict(list)
    # for task in tqdm(tasks):
    #
    #     for n_episodes in range(50, 51):
    #         for op in previous_operators:
    #             op.clear()
    #         save_dir = make_path(dir, task, 0, n_episodes)
    #
    #         env, domain, problem, n_samples, n_syms, n_ops = try_build(save_dir, task, n_episodes,
    #                                                                    previous_predicates,
    #                                                                    previous_operators)
    #         vocabulary = domain.predicates
    #
    #         for pred in vocabulary:
    #             if pred == Predicate.not_failed() or isinstance(pred, _ProblemProposition):
    #                 continue
    #             if pred.mask == [1]:
    #                 preds[task].append(pred)
    #
    # save(preds, 'syms1.pkl')
    # exit(1)
    import matplotlib.pyplot as plt
    import numpy as np

    preds = load('syms1.pkl')
    for task in range(1, 11):
        pca = PCA(PCA_STATE)
        pca.load('pca/models/dropped_key_pca_state.dat')
        pca2 = PCA(PCA_INVENTORY)
        pca2.load('pca/models/dropped_key_pca_inventory.dat')
        env = PCAWrapper(MultiTreasureGame(task, pcas=[pca, pca2], split_inventory=True), pca, pca2=pca2)
        vocabulary = preds[task]
        for pred in vocabulary:
                data = pred.sample(100)
                # print(np.mean(data, axis=0))
                print(np.mean(data))
                im = env.render_states(data, mask=pred.mask, masked=True, view=View.AGENT)
                plt.title("{}: {}".format(task, np.mean(data)))
                plt.imshow(im)
                plt.savefig(("{}-{}.png".format(task, np.mean(data))))
                plt.show()



    # save_dir = '../data_transfer/{}'.format(TASK)
    #
    # pca = PCA(PCA_STATE)
    # pca.load('pca/models/dropped_key_pca_state.dat')
    # pca2 = PCA(PCA_INVENTORY)
    # pca2.load('pca/models/dropped_key_pca_inventory.dat')
    # env = PCAWrapper(MultiTreasureGame(TASK, pcas=[pca, pca2], split_inventory=True), pca, pca2=pca2)
    #
    # previous_predicates, previous_operators = extract(previous_domain)
    # # make sure we clear out the old stuff
    # for op in previous_operators:
    #     op.clear()
    #
    # domain = load(make_path(save_dir, 'linked_domain.pkl'))
    # problem = load(make_path(save_dir, 'linked_problem.pkl'))
    #
    # domain, problem, info = build_transfer_model(env, previous_predicates, previous_operators,
    #                                              reload=True,
    #                                              save_dir=save_dir,
    #                                              n_jobs=16,
    #                                              seed=0,
    #                                              n_episodes=50,
    #                                              options_per_episode=1000,
    #                                              view=View.AGENT,
    #                                              **CONFIG[TASK],
    #                                              visualise=False,
    #                                              verbose=True)
    #
    # print(evaluate_with_network(domain, problem))

    #
    # domain = load(make_path(save_dir, 'linked_domain.pkl'))
    # problem = load(make_path(save_dir, 'linked_problem.pkl'))
    # clusterer = load(make_path(save_dir, 'quick_cluster.pkl'))
    # prop = next(x for x in problem.init if isinstance(x, _ProblemProposition))
    # start_link = int(prop.name[prop.name.index('_') + 1:])
    #
    # tasks = discover_hddl_tasks(domain, problem, start_link, verbose=True, draw=False, subgoal_method='voterank')
    # hddl = HDDLDomain(domain)
    #
    # for task in tasks:
    #     hddl.add_task(task)
    # save(hddl, make_path(save_dir, 'hddl_domain.pkl'))
    # save(hddl, make_path(save_dir, 'domain.hddl'), binary=False)
    # flat_domain = hddl.to_pddl()
    # save(flat_domain, make_path(save_dir, 'flat_domain.pkl'))
    # #
    # exit(0)
    # print(hddl)
