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

        # returns a probability distribution
        distribution = self._get_distribution(tf.concat(inputs, axis=-1))

        # if provided; shift the distribution to a desired location
        # necessary when the prediction space is not [-1., 1.]
        out_shift = self.__dict__.get('out_shift', None)
        if out_shift is not None:
            distribution = tfp.bijectors.Shift(out_shift)(distribution)

        # if provided; shift the distribution to a desired location
        # necessary when the prediction space is not [-1., 1.]
        out_scale = self.__dict__.get('out_scale', None)
        if out_scale is not None:
            distribution = tfp.bijectors.Scale(out_scale)(distribution)

        # if provided; lower bound the prediction distribution
        # prevents samples outside the range of the prediction space
        clip_below = self.__dict__.get('clip_below', None)
        if clip_below is not None:
            distribution = tfp.bijectors.Inline(
                forward_fn=lambda x: tf.maximum(x, clip_below),
                inverse_fn=lambda x: x,
                inverse_log_det_jacobian_fn=lambda y: tf.zeros_like(y),
                is_constant_jacobian=True,
                is_increasing=True,
                name='clip_below')(distribution)

        # if provided; upper bound the prediction distribution
        # prevents samples outside the range of the prediction space
        clip_above = self.__dict__.get('clip_above', None)
        if clip_above is not None:
            distribution = tfp.bijectors.Inline(
                forward_fn=lambda x: tf.minimum(x, clip_above),
                inverse_fn=lambda x: x,
                inverse_log_det_jacobian_fn=lambda y: tf.zeros_like(y),
                is_constant_jacobian=True,
                is_increasing=True,
                name='clip_above')(distribution)

        # return a transformed distribution
        return distribution

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
        shift_fn = self.__dict__.get('shift_fn', tf.tanh)
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
            loc=shift_fn(loc), scale_diag=scale_fn(scale))


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
        shift_fn = self.__dict__.get('shift_fn', lambda z: z)
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
                loc=shift_fn(loc), scale_diag=scale_fn(scale)))
