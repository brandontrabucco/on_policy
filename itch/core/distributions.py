from abc import ABC, abstractmethod
import tensorflow as tf
import tensorflow_probability as tfp


class Distribution(ABC):

    def __init__(self,
                 model,
                 **kwargs):
        """Creates a distribution base class that can be subclassed to
        build probabilistic neural networks

        Arguments:

        model: PicklingModel
            a keras model whose outputs will be passed into a
            tensorflow probability distribution"""

        self.model = model
        self.__dict__.update(kwargs)

    def get_weights(self):
        """Gets the weights of the policy which might be sent to another python
        process for sampling using the new model

        Returns:

        weights: list
            a list of variables that can be obtained by calling
            model.get_weights on self.model"""

        return self.model.get_weights()

    def set_weights(self,
                    weights):
        """Sets the weights of the policy to the provided weights,
        which are typically from another process

        Arguments:

        weights: list
            a list of variables that can be obtained by calling
            model.get_weights on a copy of self.policy"""

        self.model.set_weights(weights)

    def __call__(self,
                 *inputs):
        """Returns a tensorflow probability distribution for sampling
        and evaluating the likelihood of samples

        Arguments:

        inputs: list[tf.Tensor]
            a list of tensors that contains inputs that parameterize the
            probability distribution

        Returns:

        distribution: tfp.Distribution
            a tensorflow probability distribution that can be sampled or used
            to evaluate the likelihood of samples under"""

        return self._get_distribution(tf.concat(inputs, axis=-1))

    @abstractmethod
    def _get_distribution(self,
                          inputs):
        """Returns a tensorflow probability distribution for sampling
        and evaluating the likelihood of samples

        Arguments:

        inputs: tf.Tensor
            a tensor that contains inputs that parameterize the
            probability distribution

        Returns:

        distribution: tfp.Distribution
            a tensorflow probability distribution that can be sampled or used
            to evaluate the likelihood of samples under"""

        return NotImplemented

    @property
    def trainable_variables(self):
        return self.model.trainable_variables


class Gaussian(Distribution):

    def _get_distribution(self,
                          inputs):
        """Returns a tensorflow probability distribution for sampling
        and evaluating the likelihood of samples

        Arguments:

        inputs: tf.Tensor
            a tensor that contains inputs that parameterize the
            probability distribution

        Returns:

        distribution: tfp.Distribution
            a tensorflow probability distribution that can be sampled or used
            to evaluate the likelihood of samples under"""

        # functions for controlling the inputs to the distribution
        loc_fn = self.__dict__.get('loc_fn', tf.tanh)
        scale_fn = self.__dict__.get('scale_fn', lambda z: z)
        log_scale_fn = self.__dict__.get(
            'log_scale_fn', lambda z: tf.clip_by_value(z, -10., 3.))

        # forward pass using the model
        loc = self.model(inputs)

        # the distribution could have a fixed scale
        log_scale = self.__dict__.get('log_scale', None)

        # calculate location and scale of the distribution
        if log_scale is None:
            loc, log_scale = tf.split(loc, 2, axis=-1)

        # for numerical stability
        scale = tf.exp(log_scale_fn(log_scale))

        # create a diagonal gaussian
        return tfp.distributions.MultivariateNormalDiag(
            loc=loc_fn(loc), scale_diag=scale_fn(scale))


class TanhGaussian(Distribution):

    def _get_distribution(self,
                          inputs):
        """Returns a tensorflow probability distribution for sampling
        and evaluating the likelihood of samples

        Arguments:

        inputs: tf.Tensor
            a tensor that contains inputs that parameterize the
            probability distribution

        Returns:

        distribution: tfp.Distribution
            a tensorflow probability distribution that can be sampled or used
            to evaluate the likelihood of samples under"""

        # functions for controlling the inputs to the distribution
        loc_fn = self.__dict__.get('loc_fn', lambda z: z)
        scale_fn = self.__dict__.get('scale_fn', lambda z: z)
        log_scale_fn = self.__dict__.get(
            'log_scale_fn', lambda z: tf.clip_by_value(z, -10., 3.))

        # forward pass using the model
        loc = self.model(inputs)

        # the distribution could have a fixed scale
        log_scale = self.__dict__.get('log_scale', None)

        # calculate location and scale of the distribution
        if log_scale is None:
            loc, log_scale = tf.split(loc, 2, axis=-1)

        # for numerical stability
        scale = tf.exp(log_scale_fn(log_scale))

        # create a diagonal tanh gaussian distribution
        return tfp.bijectors.Tanh()(
            tfp.distributions.MultivariateNormalDiag(
                loc=loc_fn(loc), scale_diag=scale_fn(scale)))
