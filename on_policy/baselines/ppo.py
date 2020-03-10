from on_policy.keras_utils import PicklingSequential
from on_policy.utils.tensorboard_logger import TensorboardLogger
from on_policy.core.distributions import Gaussian
from on_policy.algorithms.ppo import PPO
from on_policy.agents.gae_agent import GaeAgent
from on_policy.data.sampler import Sampler
from on_policy.core.train import train
from on_policy.data.worker import identity
import tensorflow as tf


ppo_variant = dict(
    hidden_size=64,
    logging_dir='ppo/',
    policy_learning_rate=3e-4,
    critic_learning_rate=3e-4,
    epoch=10,
    batch_size=2000,
    epsilon=0.2,
    entropy_bonus=0.01,
    discount=0.99,
    lamb=0.95,
    obs_selector=identity,
    num_workers=10,
    max_horizon=1000,
    iterations=1000,
    steps_per_iteration=10000)


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
        tf.keras.layers.Dense(variant['hidden_size'], input_shape=(obs_size,)),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dense(variant['hidden_size']),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dense(act_size * 2)],
        name='policy')

    value_function = PicklingSequential([
        tf.keras.layers.Dense(variant['hidden_size'], input_shape=(obs_size,)),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dense(variant['hidden_size']),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dense(1)],
        name='value_function')

    logger = TensorboardLogger(variant['logging_dir'])
    policy = Gaussian(policy)

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

    sampler = Sampler(
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
