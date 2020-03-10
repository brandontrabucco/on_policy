from itch import init_process
import tensorflow as tf
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
                    self.__class__.__name__: self.__class__})

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


class PicklingSequential(Pickling, tf.keras.Sequential):

    pass


class PicklingModel(Pickling, tf.keras.Model):

    pass
