import functools
import math

import numpy as np

import hasher

class PyLSHModel:
    """
    Wrapper class for LSH model.
    """
    def __init__(self, budget, min_clusters=2, target_threshold=None, n_bands=None, seed=None):
        """
        Initialize the LSH model. Only one of the parameters target_threshold or
        n_bands is required.
        Parameters
        ----------
        budget : integer
            Total number of rows to split the signatures into.
        min_clusters : integer
            Minimum allowable cluster size.
        target_threshold: float, optional
            Value of desired threshold if bands not specified.
        n_bands : integer, optional
            Number of bands.
        """
        
        self.budget = budget # budget is the total number of rows: rows*bands
        self.target_threshold = target_threshold
        self.min_clusters = min_clusters
        self.rng = np.random.default_rng(seed)

        if n_bands:
            self.n_bands = n_bands
            self.n_rows = budget // n_bands
            self.threshold = target_threshold
        else:
            self.n_bands, self.n_rows, self.threshold = self.__tune_parameters()
        
    def __tune_parameters(self):
        for bands in divisors(self.budget, reverse=True):
            rows = self.budget // bands
            threshold = (1.0 / bands) ** (1.0 / rows)
            print(bands, rows, threshold)
            if (threshold > self.target_threshold):
                return bands, rows, threshold

    def run(self, data, p=None, m=5):
        """
        Starts the main LSH process.
        Parameters
        ----------
        data : RDD[Vector]
            RDD of data points. Acceptable vector types are numpy.ndarray,
            list or PySpark SparseVector.
        p : integer
            Prime number larger than the largest value in data.
        m : integer
            Number of bins for hashing.
        """

        zdata = data.zipWithIndex()
        n_planes = math.ceil(np.log2(m))
        planes = self.rng.choice(a=[1, -1], size=(self.n_rows, n_planes))

        hashes = functools.partial(hasher.vector_hash, planes=planes)

        # Start by generating the signatures for each data point.
        # Output format is:        
        # <(vector idx, band idx), hash>
        sigs = zdata.flatMap(lambda x: [[(x[1], i), hashes(v=x[0][i*self.n_rows:(i+1)*(self.n_rows)])] for i in range(self.n_bands)]).cache()
        
        # Put together the vector minhashes in the same band.
        # Output format is:
        # <(band idx, hash minhash-list), vector idx>
        bands = sigs.map(lambda x: [(x[0][1], x[1]), x[0][0]]) \
            .groupByKey().mapValues(list).cache()
        # print(bands.mapValues(list).collect())

        # Filter the bucket with size < min_clusters
        if self.min_clusters > 0:
            bands = bands.filter(lambda x: len(x[1]) >= self.min_clusters).cache()

        # print(bands.mapValues(list).sortByKey().collect())
        # print('--------')
        # Remaps each element to a cluster / bucket index.
        # Output format is:
        # <vector idx, bucket idx>
        vector_bucket = bands.map(lambda x: frozenset(sorted(x[1]))).distinct() \
            .zipWithIndex().flatMap(lambda x: map(lambda y: (np.long(y), x[1]), x[0])) \
            .cache()

        # Reverses indices, to key the vectors by their buckets.
        # Output format is:
        # <bucket idx, vector idx>
        bucket_vector = vector_bucket.map(lambda x: (x[1], x[0])).cache()

        # Joins indices up with original data to provide clustering results.
        # Output format is:
        # <bucket idx, list of vectors>
        buckets = zdata.map(lambda x: (x[1], x[0])).join(vector_bucket) \
            .map(lambda x: (x[1][1], x[1][0])).groupByKey().cache()

        # print(buckets.mapValues(list).collect())
        # print(bucket_vector.groupByKey().sortByKey().mapValues(sorted).collect())
        clusters = bucket_vector.groupByKey().map(lambda x: x[1]).cache()
        # print('-----')
        # print(clusters.map(sorted).collect())

        return clusters

        # Computes Jaccard similarity of each bucket.
        # scores = buckets.map(distance_metric).cache()
        
        

def divisors(n, reverse=False):
    second = []

    for i in range(1, n//2):
        if n % i == 0:
            (ans, other) = (i, n//i) if not reverse else (n//i, i)
            second.insert(0, other)
            yield ans
    
    for val in second:
        yield val
