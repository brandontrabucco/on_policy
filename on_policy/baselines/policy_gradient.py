from on_policy.utils.keras_utils import PicklingSequential
from on_policy.tensorboard_logger import TensorboardLogger
from on_policy.distributions import Gaussian
from on_policy.algorithms.policy_gradient import PolicyGradient
from on_policy.agents.policy_agent import PolicyAgent
from on_policy.data.parallel_sampler import ParallelSampler
from on_policy.core.train import train
from on_policy.data.sampler import identity
import tensorflow as tf


policy_gradient_variant = dict(
    hidden_size=64,
    logging_dir='policy_gradient/',
    policy_learning_rate=3e-4,
    epoch=10,
    batch_size=2000,
    entropy_bonus=0.01,
    discount=0.99,
    obs_selector=identity,
    num_workers=10,
    max_horizon=1000,
    iterations=1000,
    steps_per_iteration=10000)


def policy_gradient(variant,
                    env):
    """Trains a policy using Policy Gradients (REINFORCE)
    in the provided environment

    Arguments:

    variant: dict
        a dictionary of hyper parameters for controlling the training
        of the rl algorithm
    env: gym.Env
        an environment on which to train the agent"""

    obs_size = env.observation_space.low.size
    act_size = env.action_space.low.size

    policy = PicklingSequential([
        tf.keras.layers.Dense(variant['hidden_size'], input_shape=(obs_size,)),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dense(variant['hidden_size']),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dense(act_size)],
        name='policy')

    scale = (env.action_space.high - env.action_space.low) / 2
    shift = (env.action_space.high + env.action_space.low) / 2
    policy = Gaussian(policy,
                      log_scale=tf.math.log(tf.tile([[0.1]], [1, act_size])),
                      out_scale=scale[tf.newaxis],
                      out_shift=shift[tf.newaxis])

    logger = TensorboardLogger(variant['logging_dir'])

    algorithm = PolicyGradient(
        policy,
        policy_learning_rate=variant['policy_learning_rate'],
        epoch=variant['epoch'],
        batch_size=variant['batch_size'],
        entropy_bonus=variant['entropy_bonus'],
        logger=logger,
        name='policy_gradient/')

    agent = PolicyAgent(
        policy,
        discount=variant['discount'],
        obs_selector=variant['obs_selector'])

    sampler = ParallelSampler(
        env,
        agent,
        num_workers=variant['num_workers'],
        max_horizon=variant['max_horizon'])

    train(
        sampler,
        agent,
        algorithm,
        logger=logger,
        iterations=variant['iterations'],
        steps_per_iteration=variant['steps_per_iteration'])
