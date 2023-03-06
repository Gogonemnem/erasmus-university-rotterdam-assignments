import tensorflow as tf
import attention


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

    def call(self, inputs, return_weights=False):
        attention_score = self.dense(inputs)
        attention_weights = self.weighter(attention_score)
        averaged_attention_weight = self.averager(attention_weights)
        averaged_attention_weights = self.repeater(averaged_attention_weight)

        # print(inputs.shape, attention_weights.shape)
        feature_representation = inputs * attention_weights
        # print(feature_representation.shape)
        # feature_representation = inputs * attention_score

        if return_weights:
            return feature_representation, attention_weights
        
        return feature_representation


class Model(tf.keras.Model):
    def __init__(self, num_feature_prediction, out_steps):
        super().__init__()
        
        self.attention = attention.MHAFS()

        #### STILL PLAYING WITH RETURN_SEQUENCES TRUE/FALSE
        # self.lstm2 = tf.keras.layers.LSTM(6, activation='softmax', return_sequences=True)
        self.lstm = tf.keras.layers.LSTM(128, activation='softmax', return_sequences=False)
        
        self.dense10 = tf.keras.layers.Dense(10)
        self.dense = tf.keras.layers.Dense(out_steps*num_feature_prediction, activation='linear')

        self.flat = tf.keras.layers.Flatten()
        self.drop = tf.keras.layers.Dropout(0.2)
        self.reshape = tf.keras.layers.Reshape([out_steps, num_feature_prediction])
    
    def call(self, inputs):
        x = inputs
        # print(f'inputs: {x.shape}')
        x = self.attention(x)
        # print(f'attentioned: {x.shape}')
        # print(attention_weights)

        # x = self.lstm2(x)
        # x = self.flat(x)

        x = self.lstm(x)

        x = self.drop(x)
        
        # print(f'lstm: {x.shape}')
        x = self.dense10(x)
        x = self.dense(x)
        x = self.reshape(x)
        # print(f'pred: {x.shape}')
        return x
    
    def get_attention_weights(self, X):
        return self.attention(X, return_weights=True)[1]

class SingleShot(tf.keras.Model):
    def __init__(self, out_steps, number_of_features, num_layers=0, units=32, lstm_layers=None, lstm_units=None, dense_layers=None, dense_units=None, heads=None, key_dim=None, dropout=0.2, kernel_regularizer=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.lstm_units = units if lstm_units is None else lstm_units
        self.dense_units = units if dense_units is None else dense_units
        self.key_dim = units if key_dim is None else key_dim

        self.lstm_layers = num_layers if lstm_layers is None else lstm_layers
        self.dense_layers = num_layers if dense_layers is None else dense_layers

        self.attention_layer = attention.MHAFS(heads=heads, key_dim=key_dim)
        self.lstm_layers = [tf.keras.layers.LSTM(self.lstm_units, return_sequences=True, kernel_regularizer=kernel_regularizer) for _ in range(self.lstm_layers)]
        self.final_lstm_layer = tf.keras.layers.LSTM(self.lstm_units, return_sequences=False, kernel_regularizer=kernel_regularizer)
        self.dense_layers = [tf.keras.layers.Dense(self.dense_units, activation='relu', kernel_regularizer=kernel_regularizer) for _ in range(self.dense_layers)]
        self.output_layer = tf.keras.layers.Dense(out_steps*number_of_features, kernel_regularizer=kernel_regularizer)
        self.drop = tf.keras.layers.Dropout(dropout)
        self.reshape = tf.keras.layers.Reshape([out_steps, number_of_features])

    def call(self, inputs):
        x = inputs
        x = self.attention_layer(x)
        x = self.drop(x)

        for lstm_layer in self.lstm_layers:
            x = lstm_layer(x)
            x = self.drop(x)
            
        x = self.final_lstm_layer(x)
        x = self.drop(x)
        
        for dense_layer in self.dense_layers:
            x = dense_layer(x)
            x = self.drop(x)
            
        x = self.output_layer(x)

        x = self.reshape(x)

        return x
    
    def get_attention_weights(self, input):
        _, weights = self.attention_layer(input, return_weights=True)
        return weights

class AutoregressiveFeedback(tf.keras.Model):
    def __init__(self, out_steps, number_of_features, num_layers=0, units=32, lstm_layers=None, lstm_units=None, prediction_layers=None, prediction_units=None, feature_layers=None, feature_units=None, heads=None, key_dim=None, dropout=0.2, kernel_regularizer=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.out_steps = out_steps
        
        self.lstm_units = units if lstm_units is None else lstm_units
        self.prediction_units = units if prediction_units is None else prediction_units
        self.feature_units = units if feature_units is None else feature_units
        self.key_dim = units if key_dim is None else key_dim

        self.lstm_layers = num_layers if lstm_layers is None else lstm_layers
        self.prediction_layers = num_layers if prediction_layers is None else prediction_layers
        self.feature_layers = num_layers if feature_layers is None else feature_layers

        self.attention_layer = attention.MHAFS(heads=heads, key_dim=self.key_dim)
        self.attention2 = tf.keras.layers.Attention(True)
        self.lstm_cells = [tf.keras.layers.LSTMCell(self.lstm_units, kernel_regularizer=kernel_regularizer) for _ in range(self.lstm_layers)]
        last_lstm_cell = tf.keras.layers.LSTMCell(self.lstm_units)
        
        # # LSTMCell in an RNN to simplify warmup.
        self.lstm_layers = [tf.keras.layers.RNN(lstm_cell, return_state=True, return_sequences=True) for lstm_cell in self.lstm_cells] + \
                           [tf.keras.layers.RNN(last_lstm_cell, return_state=True)]
        self.prediction_layers = [tf.keras.layers.Dense(self.prediction_units, kernel_regularizer=kernel_regularizer) for _ in range(self.prediction_layers)] + \
                                 [tf.keras.layers.Dense(number_of_features, kernel_regularizer=kernel_regularizer)]
        self.feature_layers = [tf.keras.layers.Dense(self.feature_units, kernel_regularizer=kernel_regularizer) for _ in range(self.feature_layers)]

        self.drop = tf.keras.layers.Dropout(dropout)

    def build(self, input_shape):
        self.time_steps = input_shape[1]
        number_of_all_features = input_shape[-1]
        self.feature_layers += [tf.keras.layers.Dense(number_of_all_features)]
    
    def call(self, inputs, training=None):
        x = inputs # => (batch, time, features)
        x = self.attention_layer(x) # => (batch, time, features)
        x = self.drop(x)

        predictions = []

        states = [None] * len(self.lstm_layers)
        
        for i, lstm_layer in enumerate(self.lstm_layers):
            x, *states[i] = lstm_layer(x) # => (batch, lstm_units)
            x = self.drop(x)

        predictions.append(x)
        for feature_layer in self.feature_layers:
            x = feature_layer(x)
            x = self.drop(x)

        for _ in range(1, self.out_steps):
            for i, lstm_cell in enumerate(self.lstm_cells):
                x, states[i] = lstm_cell(x, states=states[i], training=training)
                x = self.drop(x)

            predictions.append(x)
            for feature_layer in self.feature_layers:
                x = feature_layer(x)
                x = self.drop(x)

        predictions = tf.stack(predictions) # => (time, batch, features)
        x = tf.transpose(predictions, [1, 0, 2]) # => (batch, time, features)
        x = self.attention2([x, x])

        # for prediction_layer in self.prediction_layers:
        #     x = prediction_layer(x)
        #     x = self.drop(x)
        x = self.prediction_layers[-1](x)
        x = self.drop(x)
        return x
    
    def get_attention_weights(self, input):
        _, weights = self.attention_layer(input, return_weights=True)
        return weights
    

class EncoderDecoder(tf.keras.Model):
    def __init__(self, out_steps, number_of_features, num_layers=0, units=32, encoder_layers=None, encoder_units=None, decoder_layers=None, decoder_units=None,  dense_layers=None, dense_units=None, heads=None, key_dim=None, dropout=0.2, kernel_regularizer=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.encoder_units = units if encoder_units is None else encoder_units
        self.decoder_units = units if decoder_units is None else decoder_units
        self.dense_units = units if dense_units is None else dense_units
        self.key_dim = units if key_dim is None else key_dim

        self.encoder_layers = num_layers if encoder_layers is None else encoder_layers
        self.decoder_layers = num_layers if decoder_layers is None else decoder_layers
        self.dense_layers = num_layers if dense_layers is None else dense_layers
        
        
        self.attention_layer = attention.MHAFS(heads=heads, key_dim=self.key_dim)
        self.encoder_layers = [tf.keras.layers.LSTM(self.encoder_units, return_sequences=True, kernel_regularizer=kernel_regularizer) for _ in range(self.encoder_layers)] + \
                              [tf.keras.layers.LSTM(self.encoder_units, return_sequences=False, kernel_regularizer=kernel_regularizer)]
        self.repeat = tf.keras.layers.RepeatVector(out_steps)
        self.decoder_layers = [tf.keras.layers.LSTM(self.decoder_units, return_sequences=True, kernel_regularizer=kernel_regularizer) for _ in range(self.decoder_layers + 1)]
        self.dense_layers = [tf.keras.layers.Dense(self.dense_units, kernel_regularizer=kernel_regularizer) for _ in range(self.dense_layers)] + \
                            [tf.keras.layers.Dense(number_of_features, kernel_regularizer=kernel_regularizer)]

        self.drop = tf.keras.layers.Dropout(dropout)
    
    def call(self, inputs, training=None, mask=None):
        x = inputs
        x = self.attention_layer(x)
        x = self.drop(x)

        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
            x = self.drop(x)

        x = self.repeat(x)

        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x)
            x = self.drop(x)
        
        for dense_layer in self.dense_layers:
            x = dense_layer(x)
            x = self.drop(x)
        return x

    def get_attention_weights(self, input):
        _, weights = self.attention_layer(input, return_weights=True)
        return weights
    