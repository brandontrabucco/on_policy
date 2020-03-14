from on_policy.baselines.ppo import ppo, ppo_variant
from on_policy.core.launch import launch
from gym.envs.mujoco.half_cheetah import HalfCheetahEnv


if __name__ == "__main__":

    env = HalfCheetahEnv()

    ppo_variant.update(dict(
        hidden_size=64,
        logging_dir='PPO/HalfCheetah/',
        policy_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        epoch=10,
        batch_size=64,
        epsilon=0.2,
        entropy_bonus=0.0,
        discount=0.99,
        lamb=0.95,
        num_workers=1,
        max_horizon=2048,
        iterations=500,
        steps_per_iteration=2048))

    launch(ppo,
           ppo_variant,
           env,
           num_seeds=1)
