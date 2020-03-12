from on_policy.baselines.ppo import ppo, ppo_variant
from on_policy.core.launch import launch
from on_policy.tune import grid_search
import gym


if __name__ == "__main__":

    env = gym.make('HalfCheetah-v2')

    search_space = dict(
        epoch=[1,
               5,
               10,
               15],
        batch_size=[1 * 512,
                    6 * 512,
                    12 * 512,
                    24 * 512])

    ppo_variant.update(dict(
        hidden_size=64,
        logging_dir='PPO/HalfCheetah/',
        policy_learning_rate=0.0003,
        critic_learning_rate=0.0003,
        epoch=None,
        batch_size=None,
        epsilon=0.2,
        entropy_bonus=0.01,
        discount=0.99,
        lamb=0.95,
        num_workers=24,
        max_horizon=512,
        iterations=100,
        steps_per_iteration=24 * 512))

    vs = grid_search(search_space, ppo_variant)

    for v in vs:
        launch(ppo,
               v,
               env,
               num_seeds=1)
