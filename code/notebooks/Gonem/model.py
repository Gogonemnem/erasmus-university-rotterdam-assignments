import tensorflow as tf


class Attention(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        
        self.weighter = tf.keras.layers.Softmax(axis=2)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise Exception("Wrong dimensions")
        
        self.batch_size, self.time_steps, self.number_of_features = input_shape

        self.averager = lambda x: tf.math.reduce_mean(x, axis=1, keepdims=True)
        self.repeater = lambda x: tf.repeat(x, repeats=self.time_steps, axis=1)
        self.dense = tf.keras.layers.Dense(self.number_of_features)

        super().build(input_shape)

    def call(self, inputs):
        attention_score = self.dense(inputs)
        attention_weights = self.weighter(attention_score)
        averaged_attention_weight = self.averager(attention_weights)
        averaged_attention_weights = self.repeater(averaged_attention_weight)

        # print(inputs.shape, attention_weights.shape)
        feature_representation = inputs * attention_weights
        # print(feature_representation.shape)
        # feature_representation = inputs * attention_score
        
        return feature_representation, attention_weights


class Model(tf.keras.Model):
    def __init__(self, num_feature_prediction):
        super().__init__()
        
        self.attention = Attention()

        #### STILL PLAYING WITH RETURN_SEQUENCES TRUE/FALSE
        self.lstm2 = tf.keras.layers.LSTM(6, activation='softmax', return_sequences=True)
        # self.lstm = tf.keras.layers.LSTM(128, activation='softmax', return_sequences=False)
        
        self.dense10 = tf.keras.layers.Dense(10)
        self.dense = tf.keras.layers.Dense(num_feature_prediction, activation='linear')

        self.flat = tf.keras.layers.Flatten()
        self.drop = tf.keras.layers.Dropout(0.2)
    
    def call(self, inputs):
        x = inputs
        # print(f'inputs: {x.shape}')
        x, attention_weights = self.attention(x)
        # print(f'attentioned: {x.shape}')
        # print(attention_weights)
        x = self.lstm2(x)
        # x = self.lstm(x)
        x = self.drop(x)
        x = self.flat(x)
        # print(f'lstm: {x.shape}')
        x = self.dense10(x)
        x = self.dense(x)
        # print(f'pred: {x.shape}')
        return x
    
    def get_attention_weights(self, X):
        return self.attention(X)[1]
