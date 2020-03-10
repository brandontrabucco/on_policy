from on_policy.baselines.ppo import ppo, ppo_variant
from on_policy.launch import launch
import gym


if __name__ == "__main__":

    env = gym.make('Ant-v2')
    ppo_variant['logging_dir'] = 'Ant/'
    ppo_variant['num_workers'] = 2
    launch(ppo,
           ppo_variant,
           env,
           num_seeds=10)
