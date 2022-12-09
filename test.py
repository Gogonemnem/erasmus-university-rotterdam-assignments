# import functools
# from lsh import PyLSHModel
# import itertools
# import tensorflow_hub as hub
# import pandas as pd
# import numpy as np
# import data
# from pyspark.sql.session import SparkSession

# spark = SparkSession.builder.appName("test").getOrCreate()
# sc = spark.sparkContext

# phrase_df = pd.DataFrame({'Phrase': {0: 'strong furniture',
#   1: 'hard wood',
#   2: 'strong chair',
#   3: 'long table',
#   4: 'thick door',
#   5: 'door handle',
#   6: 'chair cushion',
#   7: 'table surface',
#   8: 'cupboard finish',
#   9: 'computer table',
#   10: 'phone charge',
#   11: 'display brightness',
#   12: 'usb cable',
#   13: 'mobile case',
#   14: 'camera quality',
#   15: 'battery draining',
#   16: 'front camera',
#   17: 'charging adapter',
#   18: 'phone heating',
#   19: 'charging time',
#   20: 'baby cloth',
#   21: 'short trousers',
#   22: 'green belt',
#   23: 'inner wear',
#   24: 'cotton pants',
#   25: 'long sleeve shirt',
#   26: 'track pants',
#   27: 'tee shirts',
#   28: 'sweat shirt',
#   29: 'red jacket',
#   30: 'skin absorb',
#   31: 'adhesive property',
#   32: 'anti aging',
#   33: 'wash face',
#   34: 'rinse well',
#   35: 'oral hygiene',
#   36: 'easily lather',
#   37: 'hair color',
#   38: 'skin protection',
#   39: 'oily texture'}})
  
# phrase_list = phrase_df["Phrase"].values
# # print(phrase_list)

# def embed(dataset, funcs):
#     phrases = [[func(d) for d in dataset] for func in funcs]
#     use_embedder = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

#     vectors = list(map(lambda x: use_embedder(x).numpy(), phrases))
#     phrase_vectors = np.hstack(vectors)
#     # print(vectors)

#     # print(phrase_vectors.shape)
#     # phrase_vectors = use_embedder(phrase_list).numpy()
#     points, v_size = phrase_vectors.shape

#     # # points = 200
#     # # v_size = 30
#     # # phrase_vectors = np.random.choice(a=np.arange(-9, 9), size=(points, v_size))
#     # print(phrase_vectors.shape)
#     # # print(phrase_vectors)

    
#     rdd = sc.parallelize(phrase_vectors)
#     return rdd, points, v_size

# # from adalsh import PyAdaLSHModel
# # lsh = PyAdaLSHModel(k=5, n_stages=3)
# # print('hi')
# # result = lsh.run(rdd.cache(), 10)
# # print('hi')
# # print(result)

# def run_lsh(dataset, funcs):
#     rdd, points, v_size = embed(dataset, funcs)

#     lsh = PyLSHModel(budget=v_size, target_threshold=0.9, seed=12345)
#     result = lsh.run(rdd.cache(), m=points*2**3)
#     pairs = result.flatMap(lambda x: list(itertools.combinations(x, 2))).map(lambda x: tuple(sorted(x))).distinct().sortBy(lambda x: x)
#     return pairs

# def filter(pairs, dataset, brand=True, webshop=True):
#     candidates = pairs.cache()
#     if brand:
#         candidates = candidates.filter(lambda x: data.brandname(dataset[x[0]]).lower() == data.brandname(dataset[x[1]]).lower()).cache()
#     if webshop:
#         get_shop = functools.partial(data.get_info, descriptor='shop')
#         candidates = candidates.filter(lambda x: get_shop(dataset[x[0]]) != get_shop(dataset[x[1]])).cache()
#     return candidates
        

# # funcs = (data.brandname, functools.partial(data.get_info, descriptor='title')) #, 
# # pairs1 = run_lsh(dataset, funcs)

# dataset = list(data.data())
# train_ds, test_ds = data.split_data(dataset, 0)
# # # train_phrases = [data.get_info(d, 'title') for d in train_ds]
# # # test_phrases = [data.get_info(d, 'title') for d in test_ds]

# # funcs = (functools.partial(data.get_info, descriptor='title'), ) #, 

# # print("creating training candidates")
# # train_pairs = run_lsh(train_ds, funcs)
# # train_candidate = filter(train_pairs, train_ds)
# # print(train_candidate.count())
# # print(len(data.dupe_indices(train_ds)))

# # print("creating testing candidates")
# # test_pairs = run_lsh(test_ds, funcs)
# # test_candidate = filter(test_pairs, test_ds)
# # print(test_candidate.count())
# # print(len(data.dupe_indices(test_ds)))

# # true = sc.parallelize(data.duplicate_indices()).cache()
# # print(true.count())

# # intersect = true.intersection(candidate).cache()
# # print(intersect.count())

# # y = np.append(np.ones(intersect.count()), np.zeros(candidate.count() - intersect.count()))

# # import eval
# # print(len(y))
# # print(eval.pair_completeness(y, y))
# # print(eval.pair_quality(y, y))
# # print('------')
# # print(eval.f1_star(y, y))
# # print(eval.f1_score(y, y))
# # print(eval.f1_hat(y, y))

# # with open('train_candidate.npy', 'wb') as f:
# #     np.save(f, np.array(train_candidate.collect()))

# # with open('test_candidate.npy', 'wb') as f:
# #     np.save(f, np.array(test_candidate.collect()))

# with open('train_candidate.npy', 'rb') as f:
#     train_candidate = np.load(f)
#     train_candidate = tuple(map(tuple, train_candidate))
#     train_candidate = sc.parallelize(train_candidate).cache()

# with open('test_candidate.npy', 'rb') as f:
#     test_candidate = np.load(f)
#     test_candidate = tuple(map(tuple, test_candidate))
#     test_candidate = sc.parallelize(test_candidate).cache()

# # print(candidate.collect())

# # data_info = functools.partial(data.get_info, descriptor='all', return_type='str')
# # pairs = candidate.map(lambda x: (data_info(dataset[x[0]]), data_info(dataset[x[1]])))


# # # print(pairs.collect())


# # true = sc.parallelize(data.duplicate_indices()).cache()
# # print(true.count())


# # # pairs = true.map(lambda x: (data_info(dataset[x[0]]), data_info(dataset[x[1]])))

# # print(candidate.count())
# # intersect = true.intersection(candidate).cache()
# # print(intersect.count())

# # true_set = {i for i in data.duplicate_indices()}

# # y = candidate.map(lambda x: float(x in true_set) * 1.0).cache()

# # print(y.count())

# # # y = np.append(np.ones(intersect.count()), np.zeros(candidate.count() - intersect.count()))

# # # print(y)

# # # y = sc.parallelize(np.ones(len(true_set))).cache()
# # model.fit(pairs.collect(), y.collect(), batch_size=128, epochs=10)

# # pos_pairs, pos_labels = data.split_data(candidate, 1)
# # pos_ds = data.make_ds(pos_pairs.collect(), pos_labels.collect())

# # neg_pairs, neg_labels = data.split_data(candidate, 0)
# # neg_ds = data.make_ds(neg_pairs.collect(), neg_labels.collect())

# # import tensorflow as tf
# # resampled_ds = tf.data.Dataset.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5])
# # resampled_ds = resampled_ds.batch(10_000).prefetch(2)




# BATCH_SIZE = 64

# # train, test = data.split_data(candidate, seed=0)
# # train, val = data.split_data(train, seed=0)

# # print("train candidate", train_candidate.collect())
# print("creating training tfds")
# train_tfds = data.ds(train_candidate, train_ds, BATCH_SIZE, balance=True)

# print("creating testing tfds")
# val_tfds = data.ds(test_candidate, test_ds, BATCH_SIZE, repeat=False, balance=False)

# print("done")



# neg_pairs, _ = data.label(train_candidate, train_ds, 0)
# print("negpairs", neg_pairs.count())
# pos_pairs, _ = data.label(train_candidate, train_ds, 1)
# print("pospairs", pos_pairs.count())
# resampled_steps_per_epoch = np.ceil(100.0*pos_pairs.count()/BATCH_SIZE)
# # print(resampled_steps_per_epoch)

# import nn
# import eval

# print("setting up nn")
# import tensorflow as tf

# loss = tf.keras.losses.BinaryFocalCrossentropy(
#     apply_class_balancing=True,
#     alpha=0.75,
#     gamma=2.0
# )


# model = nn.NN(embedding_type=None, layers=3)
# model.compile(optimizer='adam', 
#               loss=loss, 
#               metrics=[eval.precision, eval.recall, eval.f1_score, eval.f1_star], 
#               run_eagerly=False)
# model.fit(train_tfds, 
#           epochs=30, 
#           steps_per_epoch=resampled_steps_per_epoch)

# results = model.evaluate(val_tfds,
#                          steps_per_epoch=resampled_steps_per_epoch)



import pandas as pd
df = pd.read_csv('results.txt', names=('t', 'f1', 'pq', 'pc', 'f1s', 'pr', 're', 'f1_'))
results = df.groupby('t').mean()
results.to_excel('averaged.xlsx')