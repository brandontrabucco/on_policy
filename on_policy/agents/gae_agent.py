from on_policy.agents.policy_agent import PolicyAgent
from on_policy.utils.math import discounted_sum
from on_policy.data.worker import identity


class GaeAgent(PolicyAgent):

    def __init__(self,
                 policy,
                 value_function,
                 discount=0.99,
                 lamb=0.95,
                 obs_selector=identity):
        """Creates an agent class that interfaces between a policy
        and the environment

        Arguments:

        policy: PicklingModel
            a keras model that can be serialized and sent between different
            python processes for data collecting
        value_function: PicklingModel
            a keras model that can be serialized and sent between different
            python processes for data collecting
        discount: float
            the discount factor that controls the effective
            horizon of the actor's MDP
        lamb: float
            a tunable lambda factor from Generalized Advantage
            Estimation that controls the bias and variance
            of the policy gradient
        obs_selector: callable
            a function that selects into a possible nested observation
            structure such as a dictionary"""

        super(GaeAgent, self).__init__(policy,
                                       discount=discount,
                                       obs_selector=obs_selector)
        self.value_function = value_function
        self.lamb = lamb

    def get_weights(self):
        """Gets the weights of the policy which might be sent to another
        python process for sampling using the new model

        Returns:

        weights: list
            a list of variables that can be obtained by calling
            model.get_weights on self.model"""

        return (self.policy.get_weights(),
                self.value_function.get_weights())

    def set_weights(self,
                    weights):
        """Sets the weights of the policy to the provided weights,
        which are typically from another process

        Arguments:

        weights: list
            a list of variables that can be obtained by calling
            model.get_weights on a copy of self.policy"""

        self.policy.set_weights(weights[0])
        self.value_function.set_weights(weights[1])

    def get_values(self,
                   inputs):
        """Calculates the values of the provided states using the
        value function that belongs to the agent

        Arguments:

        inputs: list[tf.Tensor]
            a list of tensors that contain observations to the
            current value function

        Returns:

        values: tf.Tensor
            values representing the discounted future return"""

        return self.value_function(self.obs_selector(inputs))[:, 0]

    def get_advantages(self,
                       rewards,
                       inputs):
        """Calculates the advantages across a single episode
        using the agent's discount factor

        Arguments:

        rewards: tf.Tensor
            a sequence of rewards that were achieved during this episode
            used to compute discounted reward-to-go
        inputs: list[tf.Tensor]
            a list of tensors that contain observations to the
            current value function

        Returns:

        values: tf.Tensor
            estimates of the current advantage function"""

        # calculate generalized advantage estimates
        val = self.get_values(inputs)
        delta_v = rewards[:-1] + self.discount * val[1:] - val[:-1]
        return discounted_sum(delta_v, self.discount * self.lamb)
