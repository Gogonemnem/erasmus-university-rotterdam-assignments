
import tensorflow as tf
import keras_tuner
import numpy as np


import pandas as pd
import openpyxl # Needed for reading excel
import pathlib

import decomposition
import models



keras_tuner.__version__


tf.config.list_logical_devices()



cwd = pathlib.Path.cwd()

# code_directory = cwd.parents[1]
code_directory = cwd #/ 'code'

bas_directory = code_directory / "notebooks" / "Bas"
gonem_directory = code_directory / "notebooks" / "Gonem"
# data_file = bas_directory / "cadeautjevoorGonemenLiza.xlsx"
example_data_directory = gonem_directory / "example_data"
data_file = gonem_directory / "MAIZE_FILTERED_2023-03-03_02-09-43.xlsx"
data_file




df = pd.read_excel(data_file, header=[0, 1], index_col=0).iloc[:-2]
# df = df.loc[:, pd.IndexSlice[:, 'Ukraine']]
df.describe()


label_columns = ['price']
label_columns = df.columns[df.columns.get_level_values(0).isin(label_columns)].tolist()
label_columns


stl = decomposition.STLDecomposer(labels=label_columns, period=12)
log = decomposition.Logger(labels=label_columns)
std = decomposition.Standardizer()
mms = decomposition.MinMaxScaler()

preproc = decomposition.Processor().add(stl).add(log).add(std)


from windower import WindowGenerator

width = 24
label_width = 6
shift = 6

w = WindowGenerator(input_width=width, label_width=label_width, shift=shift, data=df, 
                    # train_begin=0, train_end=.9, val_begin=None, val_end=.96,
                    train_begin=0, train_end=.97, val_begin=None, val_end=None,
                    # train_begin=0, train_end=.5, val_begin=None, val_end=.8,
                    test_begin=None, test_end=1., connect=True, remove_labels=True, label_columns=label_columns)
w.preprocess(preproc)
w


w.train_df.tail(5)


# all(w.train_df == w.val_df)
w.val_df.head(5)


w.val_df.tail(5)


w.test_df.head(5)


w.test_df.tail(5)


label_std = decomposition.Standardizer(mean=std.mean[w.label_columns], std=std.std[w.label_columns])
label_log = decomposition.Logger(label_indices=range(len(w.label_columns)))
# label_mms = decomposition.MinMaxScaler(min=mms.min[w.label_columns], max=mms.max[w.label_columns])
postproc = decomposition.Processor().add(label_std).add(label_log)
w.add_label_postprocess(postproc)


for example_inputs, example_labels in w.train.take(1):
    print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
    print(f'Labels shape (batch, time, features): {example_labels.shape}')
    output_features = example_labels.shape[-1]



@tf.function
def closs(x, y, a=-1):
    return (x - y)**2 + (tf.sign(x-y)+a)**6


def build_ARF(hp):
    lstm_units = hp.Int("lstm_units", min_value=32, max_value=512, step=32)
    lstm_layers = hp.Int("lstm_layers", min_value=0, max_value=10)
    prediction_units = hp.Int("prediction_units", min_value=32, max_value=512, step=32)
    prediction_layers = hp.Int("prediction_layers", min_value=0, max_value=10)
    feature_units = hp.Int("feature_units", min_value=32, max_value=512, step=32)
    feature_layers = hp.Int("feature_layers", min_value=0, max_value=10)
    
    heads = hp.Int("heads", min_value=1, max_value=16)
    dropout = hp.Float("dropout", min_value=0, max_value=1)
    key_dim = hp.Int('key_dim', min_value=16, max_value=128, step=16)
    
    l1 = hp.Float("l1", min_value=1e-7, max_value=1e-1, sampling="log")
    # l1 = 0.001
    l2 = hp.Float("l2", min_value=1e-7, max_value=1e-1, sampling="log")
    # l2 = 0.001
    kernel_regularizer = tf.keras.regularizers.L1L2(l1=l1, l2=l2)

    model = models.AutoregressiveFeedback(out_steps=label_width, number_of_features=output_features, lstm_units=lstm_units, lstm_layers=lstm_layers,
                                   prediction_units=prediction_units, prediction_layers=prediction_layers, feature_units=feature_units,
                                   feature_layers=feature_layers, key_dim=key_dim, heads=heads, dropout=dropout, kernel_regularizer=kernel_regularizer)
    learning_rate = hp.Float("learning_rate", min_value=1e-7, max_value=1e-2, sampling="log")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse', 
        metrics=['mae', 'mse', 'mape', closs]
        )

    return model


def build_SS(hp):
    lstm_units = hp.Int("lstm_units", min_value=32, max_value=512, step=32)
    lstm_layers = hp.Int("lstm_layers", min_value=0, max_value=10)
    dense_units = hp.Int("dense_units", min_value=32, max_value=512, step=32)
    dense_layers = hp.Int("dense_layers", min_value=0, max_value=10)
    

    heads = hp.Int("heads", min_value=1, max_value=16)
    dropout = hp.Float("dropout", min_value=0, max_value=1)
    key_dim = hp.Int('key_dim', min_value=16, max_value=128, step=16)
    
    l1 = hp.Float("l1", min_value=1e-7, max_value=1e-1, sampling="log")
    l2 = hp.Float("l2", min_value=1e-7, max_value=1e-1, sampling="log")
    kernel_regularizer = tf.keras.regularizers.L1L2(l1=l1, l2=l2)
    
    model = models.SingleShot(out_steps=label_width, number_of_features=output_features, lstm_units=lstm_units, lstm_layers=lstm_layers, dense_units=dense_units, dense_layers=dense_layers, key_dim=key_dim, heads=heads, dropout=dropout, kernel_regularizer=kernel_regularizer)

    learning_rate = hp.Float("learning_rate", min_value=1e-7, max_value=1e-2, sampling="log")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse', 
        metrics=['mae', 'mse', 'mape']
        )

    return model


def build_ED(hp):
    encoder_units = hp.Int("encoder_units", min_value=32, max_value=512, step=32)
    encoder_layers = hp.Int("encoder_layers", min_value=0, max_value=10)
    decoder_units = hp.Int("decoder_units", min_value=32, max_value=512, step=32)
    decoder_layers = hp.Int("decoder_layers", min_value=0, max_value=10)
    dense_units = hp.Int("dense_units", min_value=32, max_value=512, step=32)
    dense_layers = hp.Int("dense_layers", min_value=0, max_value=10)

    heads = hp.Int("heads", min_value=1, max_value=16)
    dropout = hp.Float("dropout", min_value=0, max_value=1)
    key_dim = hp.Int('key_dim', min_value=16, max_value=128, step=16)
    
    l1 = hp.Float("l1", min_value=1e-7, max_value=1e-1, sampling="log")
    l2 = hp.Float("l2", min_value=1e-7, max_value=1e-1, sampling="log")
    kernel_regularizer = tf.keras.regularizers.L1L2(l1=l1, l2=l2)

    model = models.EncoderDecoder(out_steps=label_width, number_of_features=output_features, encoder_units=encoder_units, encoder_layers=encoder_layers,
                           decoder_units=decoder_units, decoder_layers=decoder_layers, dense_units=dense_units,
                           dense_layers=dense_layers, key_dim=key_dim, heads=heads, dropout=dropout, kernel_regularizer=kernel_regularizer)

    learning_rate = hp.Float("learning_rate", min_value=1e-7, max_value=1e-2, sampling="log")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse', 
        metrics=['mae', 'mse', 'mape']
        )

    return model



tuner_arf = keras_tuner.Hyperband(
    hypermodel=build_ARF,
    objective="val_mse",
    max_epochs=200,
    factor=3,
    hyperband_iterations=1,
    executions_per_trial=3,
    seed=2023,
    max_retries_per_trial=10,
    max_consecutive_failed_trials=10,
    overwrite=False,
    directory=gonem_directory/'hp',
    project_name="ARF",
)


tuner_ss = keras_tuner.Hyperband(
    hypermodel=build_SS,
    objective="val_mse",
    max_epochs=200,
    factor=3,
    hyperband_iterations=1,
    executions_per_trial=3,
    seed=2023,
    max_retries_per_trial=10,
    max_consecutive_failed_trials=10,
    overwrite=False,
    directory=gonem_directory/'hp',
    project_name="SS",
)


tuner_ed = keras_tuner.Hyperband(
    hypermodel=build_ED,
    objective="val_mse",
    max_epochs=200,
    factor=3,
    hyperband_iterations=1,
    executions_per_trial=3,
    seed=2023,
    max_retries_per_trial=10,
    max_consecutive_failed_trials=10,
    overwrite=False,
    directory=gonem_directory/'hp',
    project_name="ED",
)


early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)


# tuner_arf.search(w.train, validation_data=w.val, callbacks=[early_stopping_callback], verbose=2)


# tuner_ss.search(w.train, validation_data=w.val, callbacks=[early_stopping_callback], verbose=2)


tuner_ed.search(w.train, validation_data=w.val, callbacks=[early_stopping_callback], verbose=2)


checkpoint_path = gonem_directory / 'hp' /'best_models'
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor='val_mse', verbose=0, save_best_only=True, save_weights_only=True)


# m_arf = tuner_arf.get_best_models(num_models=1)[0]
best_hps = tuner_ss.get_best_hyperparameters()[0]
best_hps.values


m = tuner_ss.hypermodel.build(best_hps)
m.fit(w.train, epochs=200, validation_data=w.val, callbacks=[checkpoint, early_stopping_callback], verbose=2)


m.load_weights(checkpoint_path)
m.evaluate(w.test)


val_performance = {}
performance = {}

w.test

# val_performance['1'] = m.evaluate(w.val)
for i in range(6):

    label = label_columns[i]
    print(label)
    # performance['1'] = m.evaluate(w.test)
    w.plot(m, plot_col=label, max_subplots=7)



inputs, labels, predictions, weights = [], [], [], []
for x, y in w.test.take(1):
    inputs.append(x)
    labels.append(y)
    predictions.append(m(x))
    weights.append(m.attention_layer(x, return_weights=True)[1])
    
inputs = tf.concat(inputs, axis=0)
labels = tf.concat(labels, axis=0)
weights = tf.concat(weights, axis=0)
weights = tf.reduce_mean(weights, axis=0)
predictions = tf.concat(predictions, axis=0)


np.save(example_data_directory / "inputs", inputs.numpy())
np.save(example_data_directory / "labels", labels.numpy())
np.save(example_data_directory / "weights", weights.numpy())
np.save(example_data_directory / "predictions", predictions.numpy())



inputs = tf.convert_to_tensor(np.load(example_data_directory / "inputs.npy"))
labels = tf.convert_to_tensor(np.load(example_data_directory / "labels.npy"))
weights = tf.convert_to_tensor(np.load(example_data_directory / "weights.npy"))
predictions = tf.convert_to_tensor(np.load(example_data_directory / "predictions.npy"))


print(inputs.shape, labels.shape)


for weight, column in zip(weights[0], w.train_df.columns):
    print(weight.numpy(), column)


