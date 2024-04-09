import tensorflow as tf


# Code inpiration: https://gist.github.com/ekreutz/160070126d5e2261a939c4ddf6afb642
class AFS(tf.keras.layers.Layer):
    # Dot-Product Attention for Feature Selection
    def __init__(self, use_scale=True, kernel_regularizer=None, **kwargs):
        super().__init__(**kwargs)
        self.use_scale = use_scale

        self.probabilities = tf.keras.layers.Dense(2, activation='softmax', activity_regularizer=kernel_regularizer)
    
    def build(self, query_shape, key_shape=None, value_shape=None):
        if key_shape is None:
            key_shape = query_shape
        if value_shape is None:
            value_shape = query_shape

        self.perm = list(range(len(query_shape)))
        self.perm[-2], self.perm[-1] = self.perm[-1], self.perm[-2]

        if self.use_scale:
            dim_k = tf.cast(key_shape[-1], tf.float32)
            self.scale = 1 / tf.sqrt(dim_k)
        else:
            self.scale = None
    
    def call(self, query, key, value, mask=None, return_weights=False):
        score = tf.matmul(query, key, transpose_a=True)
        if self.scale is not None:
            score *= self.scale
        
        # Apply mask to the attention scores
        if mask is not None:
            score += -1e9 * mask

        # score = self.reshaper(score)
        probabilities = self.probabilities(score)

        # Tensor shape after slicing and transposing: (batch_size, ..., 1, seq_length)
        
        weight = tf.transpose(probabilities[..., :1], perm=self.perm)
        
        feature_representation = value * weight

        if return_weights:
            return feature_representation, weight
        return feature_representation

   
# class MHAFS(tf.keras.layers.Layer):
#     def __init__(self, h=8, key_dimension=None, **kwargs):
#         super(MHAFS, self).__init__(**kwargs)
#         self.h = h
#         self.key_dimension = key_dimension

#         self.attention = AFS(True)

#     def build(self, input_shape):
#         query_shape, key_shape, value_shape = input_shape
#         self.number_of_features = value_shape[-1]
#         if self.key_dimension is None:
#             self.key_dimension = self.number_of_features

#         self.layersQ = []
#         for _ in range(self.h):
#             layer = tf.keras.layers.Dense(self.number_of_features)
#             self.layersQ.append(layer)

#         self.layersK = []
#         for _ in range(self.h):
#             layer = tf.keras.layers.Dense(self.key_dimension)
#             self.layersK.append(layer)

#     def call(self, input, return_weights=False):
#         query, key, value = input

#         q = [layer(query) for layer in self.layersQ]
#         k = [layer(key) for layer in self.layersK]
#         v = [value for _ in range(self.h)]

#         # Head is in multi-head, just like the paper
#         heads = []
#         weights = []
#         for i in range(self.h):
#             head, weight = self.attention([q[i], k[i], v[i]])
#             heads.append(head)
#             weights.append(weight)

#         # head = [self.attention([q[i], k[i], v[i]]) for i in range(self.h)]
#         pooled_weight = tf.reduce_mean(tf.stack(weights), axis=0)
#         out = value * pooled_weight

#         result = [out]
#         if return_weights:
#             result.append(pooled_weight)
#         return tuple(result)

# Code inpiration: https://machinelearningmastery.com/how-to-implement-multi-head-attention-from-scratch-in-tensorflow-and-keras

class MHAFS(tf.keras.layers.Layer):
    def __init__(self, heads=8, key_dim=None, kernel_regularizer=None, **kwargs):
        super().__init__(**kwargs)
        self.attention = AFS()  # Scaled dot product attention 
        self.heads = heads  # Number of attention heads to use
        self.key_dim = key_dim  # Dimensionality of the linearly projected queries and keys
        #self.d_v = d_v  # Dimensionality of the linearly projected values
        # self.W_v = Dense(d_v)  # Learned projection matrix for the values
        self.kernel_regularizer=kernel_regularizer

    def build(self, query_shape, key_shape=None, value_shape=None):
        if value_shape is not None:
            self.number_of_features = value_shape[-1]
        elif key_shape is not None:
            self.number_of_features = key_shape[-1]
        else: 
            self.number_of_features = query_shape[-1]

        if self.key_dim is None:
            self.key_dim = self.number_of_features
        
        self.W_q = tf.keras.layers.Dense(self.number_of_features*self.heads, activity_regularizer=self.kernel_regularizer)
        self.W_k = tf.keras.layers.Dense(self.key_dim*self.heads, activity_regularizer=self.kernel_regularizer)
        self.W_v = lambda values: tf.tile(values, multiples=(1, 1, self.heads))
        # self.W_o = tf.keras.layers.Dense(self.number_of_features) # Learned projection matrix for the multi-head output
    
    def reshape_tensor(self, x, heads, flag, last_dimension=-1):
        if flag:
            # Tensor shape after reshaping and transposing: (batch_size, heads, seq_length, -1)
            
            x = tf.reshape(x, shape=(tf.shape(x)[0], tf.shape(x)[1], heads, last_dimension))
            x = tf.transpose(x, perm=(0, 2, 1, 3))
        else:
            # Reverting the reshaping and transposing operations: (batch_size, seq_length, d_k)
            x = tf.transpose(x, perm=(0, 2, 1, 3))
            x = tf.reshape(x, shape=(tf.shape(x)[0], tf.shape(x)[1], self.key_dim))
        return x
    
    def call(self, query, key=None, value=None, mask=None, return_weights=False):
        if key is None:
            key = query
        if value is None:
            value = key
        # Rearrange the queries to be able to compute all heads in parallel
        q_reshaped = self.reshape_tensor(self.W_q(query), self.heads, True, self.number_of_features)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)
 
        # Rearrange the keys to be able to compute all heads in parallel
        k_reshaped = self.reshape_tensor(self.W_k(key), self.heads, True, self.key_dim)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)
 
        # Rearrange the values to be able to compute all heads in parallel
        v_reshaped = self.reshape_tensor(self.W_v(value), self.heads, True, self.number_of_features)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)
 
        # Compute the multi-head attention output using the reshaped queries, keys and values
        o_reshaped, weights = self.attention(q_reshaped, k_reshaped, v_reshaped, mask, return_weights=True)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)

        pooled_weight = tf.reduce_mean(weights, axis=1)
        out = value * pooled_weight
 
        # Apply one final linear projection to the output to generate the multi-head attention
        # Resulting tensor shape: (batch_size, input_seq_length, d_model)
        if return_weights:
            return out, pooled_weight
        return out
        