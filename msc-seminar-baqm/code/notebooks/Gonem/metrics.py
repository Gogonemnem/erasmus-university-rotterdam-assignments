import tensorflow as tf


@tf.function
def smape(y_true, y_pred, exponent=1):
    epsilon = 1e-6
    denom = tf.abs(y_true) + tf.abs(y_pred) + epsilon
    smape = tf.reduce_mean(tf.abs(y_pred - y_true) / denom * 2.0)
    return (smape ** exponent) * 100
    