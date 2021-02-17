from gym_multi_treasure_game.envs.pca.pca_wrapper import PCAWrapper
from gym_multi_treasure_game.envs.recordable_multi_treasure_game import RecordableMultiTreasureGame
from pyddl.hddl.hddl_domain import HDDLDomain
from pyddl.pddl.predicate import Predicate
from s2s.estimators.oc_svc import OCSupportVectorClassifier
from s2s.portable.problem_symbols import _ProblemProposition


def _record_task(env, match_task):
    pass


def _match_pre(state, obs, precondition, classifiers):
    for predicate in precondition:
        if predicate == Predicate.not_failed():
            continue

        if 'psym' in predicate.name:
            x = state
        else:
            x = obs

        if classifiers[predicate.name].probability(x) < 0.5:
            return False
    return True


def _find_match(state, obs, tasks, classifiers):
    for task in tasks:
        for method in task.methods:
            if _match_pre(state, obs, method.precondition, classifiers):
                return task
    return None


def _fit_predicates(predicates):
    classifiers = dict()
    for predicate in predicates:
        if predicate == Predicate.not_failed():
            continue

        data = predicate.sample(100)
        classifier = OCSupportVectorClassifier(predicate.mask)
        classifier.fit(data, use_mask=False)
        classifiers[predicate.name] = classifier
    return classifiers



def visualise_hierarchy(version, hddl: HDDLDomain, pcas):
    tasks = {task for task in hddl.tasks if 'Level-3' in task.name}
    env = PCAWrapper(RecordableMultiTreasureGame(version, pcas=pcas), pcas[0], pcas[1])

    classifiers = _fit_predicates(hddl.predicates)

    while len(tasks) > 0:
        state, obs = env.reset()
        env.render()

        for _ in range(1000):

            match_task = _find_match(state, obs, tasks, classifiers)
            if match_task is not None:
                next_state, next_obs, reward, done, info = _record_task(env, match_task)
                tasks.remove(match_task)
            else:
                action = env.sample_action()
                next_state, next_obs, reward, done, info = env.step(action)
            env.render()

            if done:
                break

            state = next_state
            obs = next_obs
