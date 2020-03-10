from abc import ABC, abstractmethod


class Agent(ABC):

    @abstractmethod
    def get_weights(self):
        """Gets the weights of the policy which might be sent to another python
        process for sampling using the new model

        Returns:

        weights: list
            a list of variables that can be obtained by calling
            model.get_weights on self.model"""

        return NotImplemented

    @abstractmethod
    def set_weights(self,
                    weights):
        """Sets the weights of the policy to the provided weights,
        which are typically from another process

        Arguments:

        weights: list
            a list of variables that can be obtained by calling
            model.get_weights on a copy of self.policy"""

        return NotImplemented

    @abstractmethod
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
            a tensor that contains observations to the
            probability distribution

        Returns:

        samples: tf.Tensor
            samples from the current exploration policy"""

        return NotImplemented

    @abstractmethod
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
            a tensor that contains observations to the
            probability distribution

        Returns:

        samples: tf.Tensor
            means from the current exploration policy"""

        return NotImplemented

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

    @abstractmethod
    def get_values(self,
                   observations):
        """Calculates the values of the provided states using the
        value function that belongs to the agent

        Arguments:

        observations: tf.Tensor
            a tensor that contains observations to the
            current value function

        Returns:

        values: tf.Tensor
            values representing the discounted future return"""

        return NotImplemented

    @abstractmethod
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

        return NotImplemented

    @abstractmethod
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
            a tensor that contains observations to the
            current value function

        Returns:

        values: tf.Tensor
            estimates of the current advantage function"""

        return NotImplemented
