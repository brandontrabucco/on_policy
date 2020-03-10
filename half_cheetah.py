from itch.baselines.ppo import ppo, ppo_variant
from itch.launch import launch
import gym


if __name__ == "__main__":

    env = gym.make('HalfCheetah-v2')
    ppo_variant['logging_dir'] = 'HalfCheetah/'
    ppo_variant['num_workers'] = 2
    launch(ppo,
           ppo_variant,
           env,
           num_seeds=10)
