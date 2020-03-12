from on_policy.algorithms.algorithm import Algorithm
import tensorflow as tf
import math


TensorDataset = tf.data.Dataset.from_tensor_slices


class PolicyGradient(Algorithm):

    def __init__(self,
                 policy,
                 policy_learning_rate=0.0003,
                 epoch=10,
                 batch_size=32,
                 logger=None,
                 name='policy_gradient/'):
        """Creates an optimizer using Policy Gradient for training
        a policy with reinforcement learning

        Arguments:

        policy: Distribution
            a distribution that can be called in order to
            construct the conditional distribution of the
        policy_learning_rate: float
            the learning rate passed into the policy
            optimizer for training
        epoch: int
            the number of epochs to loop over the training data
            collected for the policy and critic
        batch_size: int
            the batch size to sample from the training data
            collected for the policy and critic
        logger: TensorboardLogger
            a logger that stores training statistics; use logger.record
            to save values to the logger
        name: str
            the logging prefix to use when saving statistics in this
            algorithm to the logger"""

        super(PolicyGradient, self).__init__(logger=logger, name=name)
        self.policy = policy
        self.p_optimizer = tf.keras.optimizers.Adam(
            learning_rate=policy_learning_rate)

        self.epoch = epoch
        self.batch_size = batch_size

    def policy_loss(self,
                    observations,
                    actions,
                    advantages,
                    old_log_prob,
                    log=False):
        """Creates the loss function for the critic and
        logs training info to tensorboard

        Arguments:

        observations: tf.Tensor
            the observations experienced by the current policy that
            are shaped like [batch_size, obs_dim]
        actions: tf.Tensor
            the actions taken by the current policy that
            are shaped like [batch_size, act_dim]
        advantages: tf.Tensor
            the advantages experienced by the current policy that
            are shaped like [batch_size]
        old_log_prob: tf.Tensor
            the log probabilities of the actions under the older
            policy that took those actions
        log: bool
            determines whether to log information about training to
            the logger for visualization

        Returns:

        loss: tf.tensor
            the average loss for training the value function that
            is shaped like [batch_dim]"""

        # forward pass using the current policy
        p = self.policy(observations)

        # calculate the importance sampled weight
        is_ratio = tf.exp(p.log_prob(actions) - old_log_prob)

        # compute a surrogate training loss function
        surrogate = is_ratio * advantages

        # log to tensorboard if logger is provided
        if log and self.logger is not None:
            self.logger.record(self.name + 'observations',
                               observations)
            self.logger.record(self.name + 'actions',
                               actions)
            self.logger.record(self.name + 'advantages',
                               advantages)
            self.logger.record(self.name + 'old_log_prob',
                               old_log_prob)
            self.logger.record(self.name + 'is_ratio',
                               is_ratio)
            self.logger.record(self.name + 'surrogate',
                               surrogate)

        return -tf.reduce_mean(surrogate)

    def train(self,
              observations,
              actions,
              returns,
              advantages):
        """Trains the policy using a batch of data collected
        from the environment

        Arguments:

        observations: tf.Tensor
            a tensor containing observations experienced by the agent
            that is shaped like [batch_dim, obs_dim]
        actions: tf.Tensor
            a tensor containing actions taken by the agent
            that is shaped like [batch_dim, act_dim]
        returns: tf.Tensor
            a tensor containing returns experienced by the agent
            that is shaped like [batch_dim]
        advantages: tf.Tensor
            a tensor containing advantages estimated by the agent
            that is shaped like [batch_dim]"""

        # standardize the advantages
        advantages = advantages - tf.reduce_mean(advantages)
        advantages = advantages / tf.math.reduce_std(advantages)

        # calculate the log probability of the actions
        log_prob = tf.stop_gradient(
            self.policy(observations).log_prob(actions))

        # since we are taking many gradient steps
        # it is efficient to preload data
        dataset = TensorDataset({'observations': observations,
                                 'actions': actions,
                                 'advantages': advantages,
                                 'log_probs': log_prob})

        dataset = dataset.shuffle(observations.shape[0])
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.repeat(self.epoch)
        dataset = dataset.prefetch(2)

        final_step = math.ceil(
            observations.shape[0] * self.epoch / self.batch_size) - 1

        # train the policy for a number of epochs
        for i, batch in enumerate(dataset):

            self.p_optimizer.minimize(
                lambda: self.policy_loss(batch['observations'],
                                         batch['actions'],
                                         batch['advantages'],
                                         batch['log_probs'],
                                         log=i == final_step),
                self.policy.trainable_variables)
