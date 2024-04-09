from functools import reduce
import functools
import itertools
import json
import re
import numpy as np

# from pyspark.sql.session import SparkSession
# spark = SparkSession.builder.appName("data").getOrCreate()
# sc = spark.sparkContext

features_all = {}
path = "TVs-all-merged.json"

def data(path=path, minimum_duplicates=1, descriptor=None, feature=None):
    with open(path) as file:
        data = json.load(file)
        for product in data:
            for point in data[product]:
                if len(data[product]) >= minimum_duplicates:
                    yield get_info(point, descriptor, feature)
                else:
                    yield None

def brandname(datapoint):
    for var in ('Brand', 'Brand Name'):
        if var in datapoint['featuresMap']:
            return datapoint['featuresMap'][var]
    return datapoint['title'].split()[0]  

def getmodelIDfromTitle(datapoint):
    #empty.append( list(re.finditer(r'[a-zA-Z0-9]*(([0-9]+[^0-9, ]+)|([^0-9, ]+[0-9]+))[a-zA-Z0-9]*',  item.get("title"))))
    z = re.finditer(r'[a-zA-Z0-9]*(([0-9]+[^0-9, ]+)|([^0-9, ]+[0-9]+))[a-zA-Z0-9]*',  datapoint['title'])
    temp=[]
    for i in z:
        temp.append(i.group())
    word= max(temp, key=len).replace('1080p', '')
    word=word.replace("720p", '')
    word=word.replace("1080P", '')

    word=word.replace("600Hz", '')
    return word        

def get_info(datapoint, descriptor=None, feature=None, return_type='list'):
    if datapoint is None or (descriptor is None and feature is None):
        return datapoint
    
    result = {}
    if type(descriptor) == str:
        if descriptor == 'all':
            result = datapoint
        elif descriptor == 'featuresMap':
            for feature in datapoint[descriptor]:
                result[feature] = datapoint[descriptor][feature]
        else:
            result[descriptor] = datapoint[descriptor]
    elif descriptor is not None: # assuming iterable
        for d in descriptor:
            if d == 'featuresMap':
                for feature in datapoint[d]:
                    result[feature] = datapoint[d][feature]
            else:
                result[d] = datapoint[d]
    
    if type(feature) == str:
        if feature in datapoint['featuresMap']:
            result[feature] = datapoint['featuresMap'][feature]
    elif feature is not None and descriptor != 'featuresMap': # assuming iterable
        for f in feature:
            if f in datapoint['featuresMap']:
                result[f] = datapoint['featuresMap'][f]
    
    if len(result) == 0:
        return None

    if return_type == 'list':
        result = list(result.values())
        if len(result) == 1:
            result = result[0]
    elif return_type == 'str':
        result = str(' '.join(list(result.values())))
        # result = str(result)
    return result

def duplicate_indices(path=path, n_duplicates=2):
    dic = {}
    for i, d in enumerate(data(path, n_duplicates)):
        if d is None:
            continue

        id = get_info(d, 'modelID')
        if id not in dic:
            dic[id] = [i]
        else: 
            dic[id].append(i)
    
    flat_map = lambda f, xs: reduce(lambda a, b: a + b, map(f, xs))
    pairs = flat_map(lambda x: list(itertools.combinations(x, 2)), dic.values())
    return pairs

def dupe_indices(dataset=None):
    dic = {}
    if dataset is None:
        dataset = data(minimum_duplicates=2)

    for i, d in enumerate(dataset):
        if d is None:
            continue

        id = get_info(d, 'modelID')
        if id not in dic:
            dic[id] = [i]
        else: 
            dic[id].append(i)
    

    dic = {key:dic[key] for key in dic if len(dic[key])>1}
    flat_map = lambda f, xs: reduce(lambda a, b: a + b, map(f, xs))
    pairs = flat_map(lambda x: list(itertools.combinations(x, 2)), dic.values())
    return pairs
    
def split_data(dataset, seed=None):
    # uses bootstrap with replacement 
    # duplicate samples get rejected
    # original data not in bootstrap becomes evaluation data
    rng = np.random.default_rng(seed)
    train_indices = np.sort(np.unique(rng.choice(len(dataset)-1, len(dataset))))
    test_indices = np.setdiff1d(range(len(dataset)-1), train_indices, True)
    train = [dataset[i] for i in train_indices]
    test = [dataset[i] for i in test_indices]
    return train, test


def label(indices, dataset, label=None):
    true_set = {i for i in dupe_indices(dataset)}
    data_info = functools.partial(get_info, descriptor=["title", "featuresMap"], return_type='str')
    
    if label is None:
        predicate = lambda x: True
    elif label == 1:
        predicate = lambda x: x in true_set
    elif label == 0:
        predicate = lambda x: x not in true_set
    
    pairs = indices.filter(predicate).cache()
    
    labels = pairs.map(lambda x: 1.0 * (x in true_set)).cache()
    pairs = pairs.map(lambda x: ((getmodelIDfromTitle(dataset[x[0]]), data_info(dataset[x[0]])), (getmodelIDfromTitle(dataset[x[1]]), data_info(dataset[x[1]])))).cache()
    return pairs, labels

import tensorflow as tf

def make_ds(features, labels, shuffle=True, repeat=True):
    ds = tf.data.Dataset.from_tensor_slices((features, labels))
    if shuffle:
        ds = ds.shuffle(100000)
    if repeat:
        ds = ds.repeat()
    return ds

def ds(indices, dataset, batch_size=32, shuffle=True, repeat=True, balance=True):
    if balance:
        pos_pairs, pos_labels = label(indices, dataset, 1)
        pos_ds = make_ds(pos_pairs.collect(), pos_labels.collect(), shuffle)
        
        neg_pairs, neg_labels = label(indices, dataset, 0)
        neg_ds = make_ds(neg_pairs.collect(), neg_labels.collect(), shuffle)
        
        ds = tf.data.Dataset.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5])

        if not repeat: # there are repeat values, but it not infinite
            total = pos_pairs.count() + neg_pairs.count()
            ds = ds.take(2*total)
    else:
        pairs, labels = label(indices, dataset)
        ds = make_ds(pairs.collect(), labels.collect(), shuffle, repeat)

    ds = ds.batch(batch_size).prefetch(2)
    return ds
# print((duplicate_indices()))

# for title in data(path, descriptor='title'):
#     print(title)

#         features = data[point][0]['featuresMap']
#         shop = data[point][0]['shop']
#         for feature_key in features.keys():
#             if feature_key not in features_all:
#                 features_all[feature_key] = {}
#             if shop not in features_all[feature_key]:
#                 features_all[feature_key][shop] = 1
#             else:
#                 features_all[feature_key][shop] += 1

# print(features_all)

# print('-----------------------------------------')
# features_new = {}
# for feature in features_all:
#     if len(features_all[feature].keys()) != 1:
#         features_new[feature] = features_all[feature]

# print(features_new)
# print(sorted(features_all.values()))
# print(data[point][0]['featuresMap'].keys())

# with open(path) as file:
#     data = json.load(file)
#     for point in data:
#         print(point)
#     print(len(data))