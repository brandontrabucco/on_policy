import tensorflow as tf


def discounted_sum(terms,
                   discount_factor):
    """Computes a discounted sum of returns across the
    first axis of a vector

    Arguments:

    terms: tf.Tensor
        a vector with a single dimension that the sum is taken over
    discount_factor: float
        the factor that is cumulatively multiplied onto terms

    Returns:

    returns: tf.Tensor
        the cumulative discounted sum of returns"""

    # compute discounted sum of rewards across terms
    weights = tf.tile([discount_factor], [terms.shape[0]])
    weights = tf.math.cumprod(weights, axis=0, exclusive=True)
    return tf.math.cumsum(
        terms * weights, axis=0, reverse=True) / weights
