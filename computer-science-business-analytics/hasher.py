# -*- coding: UTF-8 -*-
#
# This file is part of PySpark AdaLSH project.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2016  Giovanni Morrone (giovanni.morrone.8@gmail.com)

import numpy as np

from pyspark.mllib.linalg import SparseVector

def minhash(v, a, b, p, m):
    """
    Determines the type and computes the minhash of the vector.
        1: Multiplies the index by the non-zero seed "a".
        2: Adds the bias "b" (can be 0).
        3: Modulo "p", a number larger than the number of elements.
        4: Modulo "m", the number of buckets.

    Parameters
    ----------
    v : object
        Python list, NumPy array, or SparseVector.
    a : integer
        Seed, > 0.
    b : integer
        Seed, >= 0.
    p : integer
        Only restriction is that this number is larger than the number of elements.
    m : integer
        Number of bins.

    Returns
    -------
    i : integer
        Integer minhash value that is in [0, buckets).
    """
    indices = None
    
    if type(v) is SparseVector:
        indices = v.indices  
    elif type(v) is np.ndarray or type(v) is list:
        indices = np.arange(len(v), dtype = np.int)
        indices = indices[v == True]
    else:
        raise Exception("Unknown array type '%s'." % type(v))

    # print(indices)
    # print(a, b, p, m)
    # print("    " , ((((a * indices) + b) % p) % m))
    # print("hello")
        
    # Map the indices to hash values and take the minimum.
    return ((((a * indices) + b) % p) % m).min()


def vector_hash(v, planes):
    #Dot vector with randomly generated planes
    # print(v)
    # print(planes)
    dot_product = np.dot(v.T, planes)
    # print(dot_product)

    # get the sign of the dot product (1,11) shaped vector
    sign_of_dot_product = np.sign(dot_product)

    h = sign_of_dot_product>=0
    # print(h)

    hash_value = 0

    n_planes = planes.shape[1]
    for i in range(n_planes):
        # increment the hash value by 2^i * h_i
        hash_value += np.power(2,i)*h[i]

    hash_value = int(hash_value)
    # print(hash_value)

    return hash_value
