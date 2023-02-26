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
    def __init__(self, out_steps, number_of_features, units=32, blocks=1, heads=None, dropout=0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.blocks = blocks
        if heads is None:
            heads = units

        self.attention_layers = [attention.MHAFS(heads=heads, key_dim=units) for _ in range(self.blocks)]
        self.lstm_layers = [tf.keras.layers.LSTM(units, activation='softmax', return_sequences=False) for _ in range(self.blocks)]
        self.dense_layers = [tf.keras.layers.Dense(out_steps*number_of_features, activation='linear') for _ in range(self.blocks)]
        # self.dense10 = tf.keras.layers.Dense(10)
        # self.dense_layer = tf.keras.layers.Dense(out_steps*number_of_features, activation='linear')
        self.drop = tf.keras.layers.Dropout(dropout)
        self.reshape = tf.keras.layers.Reshape([out_steps, number_of_features])

    def build(self, input_shape):
        return super().build(input_shape)

    def call(self, inputs):
        x = inputs

        for i in range(self.blocks):
            x = self.attention_layers[i](x)
            x = self.drop(x)
            x = self.lstm_layers[i](x)
            x = self.drop(x)
            x = self.dense_layers[i](x)
            x = self.drop(x)
            x = self.reshape(x)

        return x
    
    def get_attention_weights(self, input):
        _, weights = self.attention_layers[0](input, return_weights=True)
        return weights

class AutoregressiveFeedback(tf.keras.Model):
    def __init__(self, out_steps, number_of_features, units=32, blocks=1, heads=None, dropout=0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.out_steps = out_steps
        self.blocks = blocks
        if heads is None:
            heads = units

        self.attention_layers = [attention.MHAFS(heads=heads, key_dim=units) for _ in range(self.blocks)]
        self.lstm_cells = [tf.keras.layers.LSTMCell(units) for _ in range(self.blocks)]
        # LSTMCell in an RNN to simplify warmup.
        self.lstm_layers = [tf.keras.layers.RNN(self.lstm_cells[i], return_state=True) for i in range(self.blocks)]
        self.prediction_layers = [tf.keras.layers.Dense(units) for _ in range(self.blocks-1)] + [tf.keras.layers.Dense(number_of_features)]

        self.drop = tf.keras.layers.Dropout(dropout)

    def build(self, input_shape):
        number_of_all_features = input_shape[-1] 
        self.dense_layers = [tf.keras.layers.Dense(number_of_all_features) for _ in range(self.blocks)]
    
    def call(self, inputs, training=None):
        x = inputs # => (batch, time, features)

        for i in range(self.blocks):
            x = self.attention_layers[i](x) # => (batch, time, features)
            x = self.drop(x)
            x, *state = self.lstm_layers[i](x) # => (batch, lstm_units)
            x = self.drop(x)

            predictions = []
            full_prediction = self.dense_layers[i](x) # => (batch, features)
            predictions.append(full_prediction) # first prediction

            # rest of the prediction steps.
            for _ in range(1, self.out_steps):
                    # Use the last prediction as input.
                    x = full_prediction
                    # Execute one lstm step.
                    x, state = self.lstm_cells[i](x, states=state, training=training)
                    x = self.drop(x)
                    full_prediction = self.dense_layers[i](x)
                    x = self.drop(x)

            predictions = tf.stack(predictions) # => (time, batch, features)
            x = tf.transpose(predictions, [1, 0, 2]) # => (batch, time, features)
        
        for i in range(self.blocks):
            x = self.prediction_layers[i](x)
            x = self.drop(x)
        return x
    
    def get_attention_weights(self, input):
        _, weights = self.attention_layers[0](input, return_weights=True)
        return weights
    

class EncoderDecoder(tf.keras.Model):
    def __init__(self, out_steps, number_of_features, units=32, blocks=1, heads=None, dropout=0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.blocks = blocks
        if heads is None:
            heads = units
        
        self.blocks = blocks
        
        
        self.attention_layers = [attention.MHAFS(heads=heads, key_dim=units) for _ in range(self.blocks)]
        self.encoder_layers = [tf.keras.layers.LSTM(units, activation='softmax', return_sequences=False) for _ in range(self.blocks)]
        self.repeat = tf.keras.layers.RepeatVector(out_steps)
        self.decoder_layers =[tf.keras.layers.LSTM(units, activation='softmax', return_sequences=True) for _ in range(self.blocks)]
        dense_layers = [tf.keras.layers.Dense(number_of_features) for _ in range(self.blocks)]
        self.timed_distributed_layers = [tf.keras.layers.TimeDistributed(dense_layers[i]) for i in range(self.blocks)]

        self.drop = tf.keras.layers.Dropout(dropout)

    def build(self, input_shape):
        return super().build(input_shape)
    
    def call(self, inputs, training=None, mask=None):
        x = inputs
        for i in range(self.blocks):
            x = self.attention_layers[i](x)
            x = self.drop(x)
            x = self.encoder_layers[i](x)
            x = self.drop(x)
            x = self.repeat(x)
            x = self.decoder_layers[i](x)
            x = self.drop(x)
            x = self.timed_distributed_layers[i](x)
            x = self.drop(x)
        return x

    def get_attention_weights(self, input):
        _, weights = self.attention_layers[0](input, return_weights=True)
        return weights
    