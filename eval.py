import tensorflow as tf

import data
t = 0.7
def true_positives(y, p):
    return tf.math.reduce_sum(tf.cast(tf.math.logical_and(y==1, p>t), tf.float32))

def true_negatives(y, p):
    return tf.math.reduce_sum(tf.cast(tf.math.logical_and(y==0, p<=t), tf.float32))
    
def false_positives(y, p):
    return tf.math.reduce_sum(tf.cast(tf.math.logical_and(y==0, p>t), tf.float32))

def false_negatives(y, p):
    return tf.math.reduce_sum(tf.cast(tf.math.logical_and(y==1, p<=t), tf.float32))

def precision(y, p):
    tp = true_positives(y, p)
    fp = false_positives(y, p)
    return tf.math.divide_no_nan(tp, tp+fp)

def recall(y, p):
    tp = true_positives(y, p)
    fn = false_negatives(y, p)
    return tf.math.divide_no_nan(tp, tp+fn)

def f1_score(y, p):
    pr = precision(y, p)
    rc = recall(y, p)
    return tf.math.divide_no_nan(2*pr*rc, pr+rc)

def found_duplicates(y, p):
    return true_positives(y, p)

def n_comparisons(y):
    return tf.cast(tf.shape(y)[0], tf.float32)

def n_duplicates():
    return 1.0*len(data.dupe_indices())

def pair_quality(y, p):
    return tf.math.divide_no_nan(found_duplicates(y, p), n_comparisons(y))

def pair_completeness(y, p):
    return tf.math.divide_no_nan(found_duplicates(y, p), n_duplicates())

def f1_star(y, p):
    pq = pair_quality(y, p)
    pc = pair_completeness(y, p)
    return tf.math.divide_no_nan(2*pq*pc, pq+pc)

def f1_hat(y, p):
    f1 = f1_score(y, p)
    f1s = f1_star(y, p)
    return tf.math.divide_no_nan(2*f1*f1s, f1+f1s)
