from abc import ABC, abstractmethod


class Algorithm(ABC):

    def __init__(self,
                 logger=None,
                 name='algorithm/'):
        """Creates an optimizer using rl for training a policy
        and value function with reinforcement learning; this
        is a base class for all algorithms

        Arguments:

        logger: TensorboardLogger
            a logger that stores training statistics; use logger.record
            to save values to the logger
        name: str
            the logging prefix to use when saving statistics in this
            algorithm to the logger"""

        self.logger = logger
        self.name = name

    @abstractmethod
    def train(self,
              observations,
              actions,
              log_probs,
              returns,
              advantages):
        """Trains the policy and value function using a batch of data
        collected from the environment

        Arguments:

        observations: tf.Tensor
            a tensor containing observations experienced by the agent
            that is shaped like [batch_dim, obs_dim]
        actions: tf.Tensor
            a tensor containing actions taken by the agent
            that is shaped like [batch_dim, act_dim]
        log_probs: np.ndarray
            a tensor containing the log probabilities of the actions
            taken by the agent during a roll out
            that is shaped like [batch_dim]
        returns: tf.Tensor
            a tensor containing returns experienced by the agent
            that is shaped like [batch_dim]
        advantages: tf.Tensor
            a tensor containing advantages estimated by the agent
            that is shaped like [batch_dim]"""

        return NotImplemented
