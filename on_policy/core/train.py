import tensorflow as tf
import numpy as np
import os


def train(sampler,
          agent,
          algorithm,
          logger=None,
          iterations=1000,
          steps_per_iteration=10000,
          logging_dir='./'):
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
        the number of environment steps to collect per iteration
    logging_dir: str
        path on the disk to save miscellaneous files"""

    with sampler:

        num_steps = 0
        logger.set_step(num_steps)

        for i in range(iterations):

            eval_video = os.path.join(logging_dir, 'eval{}.mp4'.format(i))
            train_video = os.path.join(logging_dir, 'train{}.mp4'.format(i))

            sampler.set_weights(agent.get_weights())

            (observations,
             actions,
             returns,
             advantages,
             rewards,
             lengths) = sampler.sample(
                steps_per_iteration,
                deterministic=True,
                render=eval_video)

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
                render=train_video)

            logger.record('train/returns', [
                tf.reduce_sum(x)
                for x in tf.split(rewards, lengths)])

            num_steps = num_steps + np.sum(lengths)
            logger.set_step(num_steps)

            algorithm.train(observations,
                            actions,
                            returns,
                            advantages)
