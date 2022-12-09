import functools

import tensorflow as tf
from lsh import PyLSHModel
import itertools
import tensorflow_hub as hub
import pandas as pd
import numpy as np
import data
from pyspark.sql.session import SparkSession

import nn
import eval

spark = SparkSession.builder.appName("test").getOrCreate()
sc = spark.sparkContext

def embed(dataset, funcs):
    phrases = [[func(d) for d in dataset] for func in funcs]
    use_embedder = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    vectors = list(map(lambda x: use_embedder(x).numpy(), phrases))
    phrase_vectors = np.hstack(vectors)
    points, v_size = phrase_vectors.shape

    
    rdd = sc.parallelize(phrase_vectors)
    return rdd, points, v_size


def run_lsh(dataset, funcs, t):
    rdd, points, v_size = embed(dataset, funcs)

    lsh = PyLSHModel(budget=v_size, target_threshold=t)
    result = lsh.run(rdd.cache(), m=points*2**3)
    pairs = result.flatMap(lambda x: list(itertools.combinations(x, 2))).map(lambda x: tuple(sorted(x))).distinct().sortBy(lambda x: x)
    return pairs


def filter(pairs, dataset, brand=True, webshop=True):
    candidates = pairs.cache()
    if brand:
        candidates = candidates.filter(lambda x: data.brandname(dataset[x[0]]).lower() == data.brandname(dataset[x[1]]).lower()).cache()
    if webshop:
        get_shop = functools.partial(data.get_info, descriptor='shop')
        candidates = candidates.filter(lambda x: get_shop(dataset[x[0]]) != get_shop(dataset[x[1]])).cache()
    return candidates
        
def run(t):

    dataset = list(data.data())
    train_ds, test_ds = data.split_data(dataset, 0)

    funcs = (functools.partial(data.get_info, descriptor='title'), ) #, 

    print("creating training candidates")
    train_pairs = run_lsh(train_ds, funcs, t)
    train_candidate = filter(train_pairs, train_ds)

    print("creating testing candidates")
    test_pairs = run_lsh(test_ds, funcs, t)
    test_candidate = filter(test_pairs, test_ds)

    BATCH_SIZE = 64
    print("creating training tfds")
    train_tfds = data.ds(train_candidate, train_ds, BATCH_SIZE, balance=True)

    print("creating testing tfds")
    val_tfds = data.ds(test_candidate, test_ds, BATCH_SIZE, repeat=False, balance=False)


    neg_pairs, _ = data.label(train_candidate, train_ds, 0)
    # print("negpairs", neg_pairs.count())
    pos_pairs, _ = data.label(train_candidate, train_ds, 1)
    # print("pospairs", pos_pairs.count())
    resampled_steps_per_epoch = np.ceil(100.0*pos_pairs.count()/BATCH_SIZE)

    print("setting up nn")

    loss = tf.keras.losses.BinaryFocalCrossentropy(
        apply_class_balancing=True,
        alpha=0.75,
        gamma=2.0
    )


    model = nn.NN(embedding_type=None, layers=3)
    model.compile(optimizer='adam', 
                loss=loss, 
                metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), eval.f1_score], # eval.precision, eval.recall, 
                run_eagerly=False)

    model.fit(train_tfds, 
            epochs=15, 
            steps_per_epoch=resampled_steps_per_epoch,
            use_multiprocessing=True,)

    results = model.evaluate(val_tfds, batch_size=BATCH_SIZE)
    print("done")

    pos_pairs, _ = data.label(test_candidate, test_ds, 1)
    fd = pos_pairs.count()
    n_comp = test_candidate.count()
    n_dup = len(data.dupe_indices(test_ds))

    pq = fd / n_comp
    pc = fd / n_dup
    f1s = (2*pq*pc) / (pq+pc)
    f1 = (2*results[0]*results[1]) / (results[0]+results[1])
    return t, results[3], pq, pc, f1s, results[0], results[1], f1 


if __name__ == '__main__':
    ts = (0, 0.002, 0.07, 0.3, 0.6, 0.81, 0.92, 0.97, 0.99, 0.998)
    
    for t in ts:
        for i in range(5):
            with open('results1.txt', mode='a') as f:
                f.write(f'{run(t)}\n')
    # run(0.9)/
