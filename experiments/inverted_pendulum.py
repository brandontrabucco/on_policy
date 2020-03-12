from on_policy.baselines.policy_gradient import policy_gradient, policy_gradient_variant
from on_policy.core.launch import launch
import gym


if __name__ == "__main__":

    env = gym.make('InvertedPendulum-v2')
    policy_gradient_variant.update(dict(
        hidden_size=64,
        logging_dir='PG/InvertedPendulum/',
        policy_learning_rate=0.0003,
        epoch=1,
        batch_size=24 * 256,
        exploration_noise=0.2,
        discount=0.99,
        num_workers=4,
        max_horizon=256,
        iterations=1000,
        steps_per_iteration=24 * 256))
    launch(policy_gradient,
           policy_gradient_variant,
           env,
           num_seeds=6)
