from on_policy.baselines.ppo import ppo, ppo_variant
from on_policy.core.launch import launch
import gym


if __name__ == "__main__":

    env = gym.make('InvertedPendulum-v2')
    ppo_variant.update(dict(
        hidden_size=64,
        logging_dir='PPO/InvertedPendulum/',
        policy_learning_rate=0.0003,
        critic_learning_rate=0.0003,
        epoch=10,
        batch_size=6 * 256,
        epsilon=0.2,
        entropy_bonus=0.01,
        discount=0.99,
        lamb=0.95,
        num_workers=4,
        max_horizon=256,
        iterations=1000,
        steps_per_iteration=24 * 256))
    launch(ppo,
           ppo_variant,
           env,
           num_seeds=6)
