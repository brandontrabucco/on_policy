from on_policy.agents.agent import Agent
from on_policy.utils.math import discounted_sum
from on_policy.data.sampler import identity
import tensorflow as tf


class PolicyAgent(Agent):

    def __init__(self,
                 policy,
                 discount=0.99,
                 obs_selector=identity):
        """Creates an agent class that interfaces between a policy
        and the environment

        Arguments:

        policy: PicklingModel
            a keras model that can be serialized and sent between different
            python processes for data collecting
        discount: float
            the discount factor that controls the effective
            horizon of the actor's MDP
        obs_selector: callable
            a function that selects into a possible nested observation
            structure such as a dictionary"""

        self.policy = policy
        self.discount = discount
        self.obs_selector = obs_selector

    def get_weights(self):
        """Gets the weights of the policy which might be sent to another python
        process for sampling using the new model

        Returns:

        weights: list
            a list of variables that can be obtained by calling
            model.get_weights on self.model"""

        return self.policy.get_weights()

    def set_weights(self,
                    weights):
        """Sets the weights of the policy to the provided weights,
        which are typically from another process

        Arguments:

        weights: list
            a list of variables that can be obtained by calling
            model.get_weights on a copy of self.policy"""

        self.policy.set_weights(weights)

    def sample(self,
               time_step,
               observations):
        """Samples from a tensorflow probability distribution defined
        by the current policy

        Arguments:

        time_step: tf.Tensor
            the current time step of the simulation that is necessary
            for constructing delayed hierarchies
        observations: tf.Tensor
            a tensor that contains observations experienced by the agent
            during data collection

        Returns:

        samples: tf.Tensor
            samples from the current exploration policy"""

        samples = self.policy(self.obs_selector(observations)).sample()
        if samples.dtype == tf.float32:
            samples = tf.clip_by_value(samples, -1., 1.)
        return samples

    def expected_value(self,
                       time_step,
                       observations):
        """Samples the mean from a tensorflow probability distribution
        defined by the current policy

        Arguments:

        time_step: tf.Tensor
            the current time step of the simulation that is necessary
            for constructing delayed hierarchies
        observations: tf.Tensor
            a tensor that contains observations experienced by the agent
            during data collection

        Returns:

        samples: tf.Tensor
            means from the current exploration policy"""

        samples = self.policy(self.obs_selector(observations)).mean()
        if samples.dtype == tf.float32:
            samples = tf.clip_by_value(samples, -1., 1.)
        return samples

    def get_rewards(self,
                    rewards,
                    observations,
                    actions):
        """Calculates the values of the provided states using the
        value function that belongs to the agent

        Arguments:

        rewards: tf.Tensor
            a tensor that contains the rewards from teh environment that
            the agent experienced
        observations: tf.Tensor
            a tensor that contains observations experienced by the agent
            during data collection
        actions: tf.Tensor
            a tensor that contains actions taken by the agent during
            data collection

        Returns:

        values: tf.Tensor
            values representing the discounted future return"""

        return rewards

    def get_values(self,
                   observations):
        """Calculates the values of the provided states using the
        value function that belongs to the agent

        Arguments:

        observations: tf.Tensor
            a tensor that contains observations experienced by the agent
            during data collection

        Returns:

        values: tf.Tensor
            values representing the discounted future return"""

        return tf.zeros([self.obs_selector(observations).shape[0]], tf.float32)

    def get_returns(self,
                    rewards):
        """Calculates the discounted sum of returns across a single
        episode using the agent's discount factor

        Arguments:

        rewards: tf.Tensor
            a sequence of rewards that were achieved during this episode
            used to compute discounted reward-to-go

        Returns:

        returns: tf.Tensor
            means from the current exploration policy"""

        return discounted_sum(rewards, self.discount)[:-1]

    def get_advantages(self,
                       rewards,
                       observations):
        """Calculates the advantages across a single episode
        using the agent's discount factor

        Arguments:

        rewards: tf.Tensor
            a sequence of rewards that were achieved during this episode
            used to compute discounted reward-to-go
        observations: tf.Tensor
            a tensor that contains observations experienced by the agent
            during data collection

        Returns:

        values: tf.Tensor
            estimates of the current advantage function"""

        return self.get_returns(rewards)
