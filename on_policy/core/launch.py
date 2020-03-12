from on_policy import init_process
from copy import deepcopy
import multiprocessing
import datetime
import os
import tensorflow as tf
import pickle as pkl


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

        # modify the path to be unique using the local time
        date = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        variant["logging_dir"] = os.path.join(
            variant["logging_dir"], "{}/".format(date))

        # save the hyper parameters used to the disk for safe keeping
        tf.io.gfile.makedirs(variant["logging_dir"])
        with tf.io.gfile.GFile(os.path.join(
                variant["logging_dir"], 'variant.pkl'), "wb") as f:
            pkl.dump(variant, f)

        # run the experiment in the current process
        return baseline(variant, env)

    # launch the experiments on the local machine
    processes = []
    for seed in range(num_seeds):

        # modify the path to be unique using the local time
        seed_variant = deepcopy(variant)
        date = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        seed_variant["logging_dir"] = os.path.join(
            seed_variant["logging_dir"], "{}/{}/".format(seed, date))

        # save the hyper parameters used to the disk for safe keeping
        tf.io.gfile.makedirs(seed_variant["logging_dir"])
        with tf.io.gfile.GFile(os.path.join(
                seed_variant["logging_dir"], 'variant.pkl'), "wb") as f:
            pkl.dump(seed_variant, f)

        # create a process for running the experiment
        processes.append(
            multiprocessing.Process(
                target=baseline,
                args=(seed_variant, env)))

    # start running the experiments and wait for all to finish
    for p in processes:
        p.start()

        if not parallel:
            p.join()

    if parallel:
        for p in processes:
            p.join()
