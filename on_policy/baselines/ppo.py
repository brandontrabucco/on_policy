from on_policy.utils.keras_utils import PicklingSequential, LogScale
from on_policy.tensorboard_logger import TensorboardLogger
from on_policy.distributions import Gaussian
from on_policy.algorithms.ppo import PPO
from on_policy.agents.gae_agent import GaeAgent
from on_policy.data.parallel_sampler import ParallelSampler
from on_policy.core.train import train
from on_policy.data.sampler import identity
import tensorflow as tf


ppo_variant = dict(
    hidden_size=64,
    logging_dir='ppo/',
    policy_learning_rate=3e-4,
    critic_learning_rate=3e-4,
    epoch=10,
    batch_size=64,
    epsilon=0.2,
    entropy_bonus=0.0,
    discount=0.99,
    lamb=0.95,
    obs_selector=identity,
    num_workers=1,
    max_horizon=2048,
    iterations=500,
    steps_per_iteration=2048)


def ppo(variant,
        env):
    """Trains a policy using Proximal Policy Optimization
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
        tf.keras.layers.Dense(
            variant['hidden_size'], input_shape=(obs_size,)),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dense(variant['hidden_size']),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dense(act_size),
        LogScale((act_size,))],
        name='policy')

    value_function = PicklingSequential([
        tf.keras.layers.Dense(
            variant['hidden_size'], input_shape=(obs_size,)),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dense(variant['hidden_size']),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dense(1)],
        name='value_function')

    out_scale = (env.action_space.high - env.action_space.low) / 2
    out_shift = (env.action_space.high + env.action_space.low) / 2
    clip_below = env.action_space.low
    clip_above = env.action_space.high

    policy = Gaussian(
        policy,
        out_scale=out_scale[tf.newaxis],
        out_shift=out_shift[tf.newaxis],
        clip_below=clip_below[tf.newaxis],
        clip_above=clip_above[tf.newaxis])

    logger = TensorboardLogger(variant['logging_dir'])

    algorithm = PPO(
        policy,
        value_function,
        policy_learning_rate=variant['policy_learning_rate'],
        critic_learning_rate=variant['critic_learning_rate'],
        epoch=variant['epoch'],
        batch_size=variant['batch_size'],
        epsilon=variant['epsilon'],
        entropy_bonus=variant['entropy_bonus'],
        logger=logger,
        name='ppo/')

    agent = GaeAgent(
        policy,
        value_function,
        discount=variant['discount'],
        lamb=variant['lamb'],
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
        steps_per_iteration=variant['steps_per_iteration'],
        logging_dir=variant['logging_dir'])
