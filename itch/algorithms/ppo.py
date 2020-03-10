from itch.algorithms.algorithm import Algorithm
import tensorflow as tf
import math


TensorDataset = tf.data.Dataset.from_tensor_slices


class PPO(Algorithm):

    def __init__(self,
                 policy,
                 value_function,
                 policy_learning_rate=0.0003,
                 critic_learning_rate=0.0003,
                 epoch=10,
                 batch_size=32,
                 epsilon=0.2,
                 entropy_bonus=0.01,
                 logger=None,
                 name='ppo/'):
        """Creates an optimizer using PPO for training a policy
        and value function with reinforcement learning

        Arguments:

        policy: Distribution
            a distribution that can be called in order to
            construct the conditional distribution of the
            model outputs given inputs
        value_function: Distribution
            a distribution that can be called in order to
            construct the conditional distribution of the
            model outputs given inputs
        policy_learning_rate: float
            the learning rate passed into the policy
            optimizer for training
        critic_learning_rate: float
            the learning rate passed into the critic
            optimizer for training
        epoch: int
            the number of epochs to loop over the training data
            collected for the policy and critic
        batch_size: int
            the batch size to sample from the training data
            collected for the policy and critic
        epsilon: float
            a parameter that controls the clipping trust region when
            building the surrogate PPO objective
        entropy_bonus: float
            the weight given to the entropy bonus added to the PPO
            surrogate training objective
        logger: TensorboardLogger
            a logger that stores training statistics; use logger.record
            to save values to the logger
        name: str
            the logging prefix to use when saving statistics in this
            algorithm to the logger"""

        super(PPO, self).__init__(logger=logger, name=name)
        self.policy = policy
        self.value_function = value_function

        self.p_optimizer = tf.keras.optimizers.Adam(
            learning_rate=policy_learning_rate)
        self.v_optimizer = tf.keras.optimizers.Adam(
            learning_rate=critic_learning_rate)

        self.epoch = epoch
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.entropy_bonus = entropy_bonus

    def critic_loss(self,
                    observations,
                    returns,
                    log=False):
        """Creates the loss function for the critic and
        logs training info to tensorboard

        Arguments:

        observations: tf.Tensor
            the observations experienced by the current policy that
            are shaped like [batch_size, obs_dim]
        returns: tf.Tensor
            the returns experienced by the current policy that
            are shaped like [batch_size]
        log: bool
            determines whether to log information about training to
            the logger for visualization

        Returns:

        loss: tf.tensor
            the average loss for training the value function that
            is shaped like [batch_dim]"""

        # forward pass using the current value function
        values = self.value_function(observations)[:, 0]
        loss = tf.keras.losses.mean_squared_error(
            returns[:, tf.newaxis], values[:, tf.newaxis])

        # log to tensorboard if logger is provided
        if log and self.logger is not None:
            self.logger.record(self.name + 'returns',
                               returns)
            self.logger.record(self.name + 'values',
                               values)
            self.logger.record(self.name + 'value_function_loss',
                               loss)

        return tf.reduce_mean(loss)

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
        clipped_is_ratio = tf.clip_by_value(
            is_ratio, 1. - self.epsilon, 1. + self.epsilon)

        # compute a clipped surrogate training loss
        entropy = p.entropy()
        surrogate = tf.minimum(is_ratio * advantages,
                               clipped_is_ratio * advantages)

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
            self.logger.record(self.name + 'clipped_is_ratio',
                               clipped_is_ratio)
            self.logger.record(self.name + 'entropy',
                               entropy)
            self.logger.record(self.name + 'surrogate',
                               surrogate)

        return -tf.reduce_mean(
            self.entropy_bonus * entropy + surrogate)

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

        # standardize the advantages
        advantages = advantages - tf.reduce_mean(advantages)
        advantages = advantages / tf.math.reduce_std(advantages)

        # since we are taking many gradient steps
        # it is efficient to preload data
        dataset = TensorDataset({'observations': observations,
                                 'actions': actions,
                                 'returns': returns,
                                 'advantages': advantages,
                                 'log_probs': log_probs})
        dataset = dataset.shuffle(observations.shape[0])
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.repeat(self.epoch)
        dataset = dataset.prefetch(2)

        final_step = math.ceil(
            observations.shape[0] * self.epoch / self.batch_size) - 1

        # train the policy and critic for a number of epochs
        for i, batch in enumerate(dataset):

            # train the policy first, which is the same as
            # https://github.com/openai/spinningup/blob/master/
            # spinup/algos/tf1/ppo/ppo.py#L231
            self.p_optimizer.minimize(
                lambda: self.policy_loss(batch['observations'],
                                         batch['actions'],
                                         batch['advantages'],
                                         batch['log_probs'],
                                         log=i == final_step),
                self.policy.trainable_variables)

            # training the critic second should not make a difference
            self.v_optimizer.minimize(
                lambda: self.critic_loss(batch['observations'],
                                         batch['returns'],
                                         log=i == final_step),
                self.value_function.trainable_variables)
