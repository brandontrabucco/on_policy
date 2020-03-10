from itch import init_process
from copy import deepcopy
import multiprocessing


def launch(baseline,
           variant,
           env,
           num_seeds=2,
           parallel=True):
    """Launches many experiments on the local machine
    and keeps track of various seeds

    Arguments:

    baseline: callable
        the rl algorithm as a function that accepts an
        environment and a variant
    variant: dict
        a dictionary of hyper parameters that control the
        rl algorithm
    env: gym.Env
        an environment that inherits from gym.Env; note
        the environment must be serialize able
    num_seeds: int
        the total number of identical experiments to run;
        with different random seeds
    parallel: bool
        determines whether to run experiments in the background
        at the same time or one at a time"""

    # initialize tensorflow and the multiprocessing interface
    init_process()

    # if only one seed; then run in the main thread
    if num_seeds == 1:
        return baseline(variant, env)

    # launch the experiments on the local machine
    processes = []
    for seed in range(num_seeds):
        seed_variant = deepcopy(variant)
        seed_variant["logging_dir"] += "{}/".format(seed)
        processes.append(
            multiprocessing.Process(
                target=baseline,
                args=(seed_variant, env)))

    # start running the experiments
    for p in processes:
        p.start()

        if not parallel:
            p.join()
    if parallel:
        for p in processes:
            p.join()
