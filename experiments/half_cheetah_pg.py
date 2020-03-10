from on_policy.baselines.policy_gradient import policy_gradient, policy_gradient_variant
from on_policy.core.launch import launch
import gym


if __name__ == "__main__":

    env = gym.make('HalfCheetah-v2')
    policy_gradient_variant.update(dict(
        hidden_size=64,
        logging_dir='PG/HalfCheetah/',
        policy_learning_rate=3e-4,
        epoch=1,
        batch_size=2048,
        entropy_bonus=0.01,
        discount=0.99,
        num_workers=1,
        max_horizon=2048,
        iterations=500,
        steps_per_iteration=2048))
    launch(policy_gradient,
           policy_gradient_variant,
           env,
           num_seeds=1)
