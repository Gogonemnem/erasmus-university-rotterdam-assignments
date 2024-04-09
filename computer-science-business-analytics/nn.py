import functools

import tensorflow as tf
import tensorflow_hub as hub

import embedding

class NN(tf.keras.Model):
    def __init__(self, embedding_type=None, units=32, layers=2, layer_type='lstm'):
        super().__init__()

        if embedding_type is None:
            self.embedding = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
        else:
            self.embedding = embedding.BERTEmbedding(embedding_type)

        if embedding_type is not None and embedding_type > 0:
            
            layer = tf.keras.layers.LSTM
            self.last_lstm = tf.keras.layers.LSTM(units)
            self.main_layers = [tf.keras.layers.Bidirectional(layer(units, return_sequences=True)) for _ in range(layers)]
        else:
            layer = tf.keras.layers.Dense
            self.main_layers = [layer(units) for _ in range(layers)]
            self.last_lstm = None
 
        # if layer_type.lower() == 'dense':
        #     layer = tf.keras.layers.Dense
        # elif layer_type.lower() == 'lstm':
        #     layer = tf.keras.layers.LSTM
        # else:
        #     raise Exception("The type of layer is undefined")

        # self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
        # self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)

        
        
        self.sim = tf.keras.layers.Dense(units)
        self.prob = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
        self.dropout_layer = tf.keras.layers.Dropout(rate=0.2)

    def call(self, inputs, training=None):
        input1 = inputs[:, 0, 1]
        input2 = inputs[:, 1, 1]
        id1 = inputs[:, 0, 0]
        id2 = inputs[:, 0, 0]

        embedding1 = self.embedding(input1)
        embedding2 = self.embedding(input2)

        emid1 = self.embedding(id1)
        emid2 = self.embedding(id2)

        x1 = self.run_layers(embedding1, training)
        x2 = self.run_layers(embedding2, training)
        y1 = self.run_layers(emid1, training)
        y2 = self.run_layers(emid2, training)

        # x = tf.concat([x1, x2], axis=-1)
        # y = tf.concat([y1, y2], axis=-1)
        x = self.sim(tf.concat([x1, x2], axis=-1))
        y = self.sim(tf.concat([y1, y2], axis=-1))
        z = tf.concat([x, y], axis=-1)
        output = self.prob(z)
        
        print(z)
        return output

    def run_layers(self, x, training=None):
        for layer in self.main_layers:
            x = layer(x)
            x = self.dropout_layer(x, training=training)
        if self.last_lstm is not None:
            x = self.last_lstm(x)
            x = self.dropout_layer(x, training=training)
        return x