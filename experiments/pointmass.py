from on_policy.baselines.ppo import ppo, ppo_variant
from on_policy.core.launch import launch
from on_policy.envs.pointmass_env import PointmassEnv


if __name__ == "__main__":

    env = PointmassEnv()
    ppo_variant['logging_dir'] = 'Pointmass/'
    ppo_variant['num_workers'] = 2
    launch(ppo,
           ppo_variant,
           env,
           num_seeds=5)
