from itch.baselines.ppo import ppo, ppo_variant
from itch.launch import launch
from itch.pointmass_env import PointmassEnv


if __name__ == "__main__":

    env = PointmassEnv()
    ppo_variant['logging_dir'] = 'Pointmass/'
    ppo_variant['num_workers'] = 2
    launch(ppo,
           ppo_variant,
           env,
           num_seeds=5)
