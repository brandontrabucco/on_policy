from on_policy import init_process
import tensorflow as tf
import numpy as np
import tempfile


class Pickling(object):

    def __init__(self,
                 *args,
                 **kwargs):
        """Creates a wrapper for a keras model so that the model can
        be pickled and sent between processes"""

        assert isinstance(self, tf.keras.Model)
        super(Pickling, self).__init__(*args, **kwargs)

    def __getstate__(self):
        """Returns the model in a binary string format that can be
        loaded back in another process"""

        with tempfile.NamedTemporaryFile(suffix='.hdf5',
                                         delete=True) as fd:

            tf.keras.models.save_model(self,
                                       fd.name,
                                       overwrite=True)

            return dict(model_str=fd.read())

    def __setstate__(self,
                     state):
        """Loads an existing model from a binary string that was
        serialized from another python process

        Arguments:

        state: dict
            a dictionary containing the model string, which is a serialized
            binary file containing the trained model"""

        init_process()

        with tempfile.NamedTemporaryFile(suffix='.hdf5',
                                         delete=True) as fd:

            fd.write(state['model_str'])
            fd.flush()

            loaded_model = tf.keras.models.load_model(
                fd.name, custom_objects={
                    self.__class__.__name__: self.__class__,
                    ConstantLogScale.__name__: ConstantLogScale})

        self.__dict__.update(loaded_model.__dict__.copy())

    @classmethod
    def from_config(cls,
                    *args,
                    custom_objects=None,
                    **kwargs):
        """Creates the model from an hdf specification format that encodes
        custom layers and their weights"""

        custom_objects = custom_objects or {}
        custom_objects[cls.__name__] = cls
        custom_objects['tf'] = tf

        return super(Pickling, cls).from_config(
            *args, custom_objects=custom_objects, **kwargs)


class ConstantLogScale(tf.keras.layers.Layer):

    def __init__(self, output_shape, **kwargs):
        """Creates a layer for concatenating a trainable variable
        to the end of a tensor; used for diagonal covariance

        Arguments:

        output_shape: tuple
            the incoming shape of the tensor from the previous layer
            without a batch dimension"""

        super(ConstantLogScale, self).__init__(**kwargs)
        # this is so the entropy of the gaussian is -N at initialization
        # empirical according to: https://arxiv.org/pdf/1812.05905.pdf
        init = tf.constant_initializer(-0.5 * (3.0 + np.log(2.0 * np.pi)))
        self.log_scale = self.add_weight(name='log_scale',
                                         shape=(1, *output_shape),
                                         initializer=init,
                                         trainable=True)

    def call(self, x):
        """Concatenate a trainable variable to the end of a tensor from
        the previous layer; used to learn diagonal covariance

        Arguments:

        x: tf.Tensor
            a tensor from a keras model

        Returns:

        y: tf.Tensor
            the same shape as x but the last dimension is doubled"""

        return tf.concat([
            x, tf.broadcast_to(self.log_scale, tf.shape(x))], -1)

    def get_config(self):
        """Creates a parameter dictionary for instantiating a layer
        with the same shape in another python process

        Returns:

        config: dict
            a dictionary of arguments to the class; for LogScale this
            includes output_shape"""

        config = super(ConstantLogScale, self).get_config()
        config.update({'output_shape': self.log_scale.shape[1:]})
        return config


class PicklingSequential(Pickling, tf.keras.Sequential):

    pass


class PicklingModel(Pickling, tf.keras.Model):

    pass
