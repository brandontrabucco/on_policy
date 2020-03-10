import tensorflow as tf
import numpy as np


def train(sampler,
          agent,
          algorithm,
          logger=None,
          iterations=1000,
          steps_per_iteration=10000):
    """Trains an on policy rl algorithm by sampling a dataset
    and training an agent iteratively

    Arguments:

    sampler: Sampler
        a parallel dataset sampling class
    agent: Agent
        an agent that maps observations into actions
    algorithm: Algorithm
        an algorithm that implements a train function
    logger: Logger
        an optional class for logging data
    iterations: int
        the number of on policy iterations to collect data for
    steps_per_iteration: int
        the number of environment steps to collect per iteration"""

    num_steps = 0
    logger.set_step(num_steps)

    with sampler:
        for i in range(iterations):

            sampler.set_weights(agent.get_weights())

            (observations,
             actions,
             returns,
             advantages,
             rewards,
             lengths) = sampler.sample(
                steps_per_iteration,
                deterministic=True,
                render=False)

            logger.record('eval/returns', [
                tf.reduce_sum(x)
                for x in tf.split(rewards, lengths)])

            (observations,
             actions,
             returns,
             advantages,
             rewards,
             lengths) = sampler.sample(
                steps_per_iteration,
                deterministic=False,
                render=False)

            logger.record('train/returns', [
                tf.reduce_sum(x)
                for x in tf.split(rewards, lengths)])

            num_steps = num_steps + np.sum(lengths)
            logger.set_step(num_steps)

            algorithm.train(observations,
                            actions,
                            returns,
                            advantages)
