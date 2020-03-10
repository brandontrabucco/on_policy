import multiprocessing as m
import tensorflow as tf


PROCESS_IS_INITIALIZED = False


def init_process():

    global PROCESS_IS_INITIALIZED
    if not PROCESS_IS_INITIALIZED:
        PROCESS_IS_INITIALIZED = True

        # set tensorflow to disable logging warnings
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

        # on startup ensure all processes are started using the spawn method
        # see https://github.com/tensorflow/tensorflow/issues/5448
        m.set_start_method('spawn', force=True)

        # prevent any process from consuming all gpu memory
        for gpu in tf.config.experimental.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
