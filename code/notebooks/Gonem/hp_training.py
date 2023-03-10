import tensorflow as tf
import keras_tuner

import models
from metrics import smape


def build_ARF(hp, out_steps, number_of_features):
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

    model = models.AutoregressiveFeedback(out_steps=out_steps, number_of_features=number_of_features, lstm_units=lstm_units, lstm_layers=lstm_layers,
                                   prediction_units=prediction_units, prediction_layers=prediction_layers, feature_units=feature_units,
                                   feature_layers=feature_layers, key_dim=key_dim, heads=heads, dropout=dropout, kernel_regularizer=kernel_regularizer)
    learning_rate = hp.Float("learning_rate", min_value=1e-7, max_value=1e-2, sampling="log")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=smape, 
        metrics=['mae', 'mse', 'mape', smape]
        )

    return model


def build_SS(hp, out_steps, number_of_features):
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
    
    model = models.SingleShot(out_steps=out_steps, number_of_features=number_of_features, lstm_units=lstm_units, lstm_layers=lstm_layers, dense_units=dense_units, dense_layers=dense_layers, key_dim=key_dim, heads=heads, dropout=dropout, kernel_regularizer=kernel_regularizer)

    learning_rate = hp.Float("learning_rate", min_value=1e-7, max_value=1e-2, sampling="log")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=smape, 
        metrics=['mae', 'mse', 'mape', smape]
        )

    return model

def build_ED(hp, out_steps, number_of_features):
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

    model = models.EncoderDecoder(out_steps=out_steps, number_of_features=number_of_features, encoder_units=encoder_units, encoder_layers=encoder_layers,
                           decoder_units=decoder_units, decoder_layers=decoder_layers, dense_units=dense_units,
                           dense_layers=dense_layers, key_dim=key_dim, heads=heads, dropout=dropout, kernel_regularizer=kernel_regularizer)

    learning_rate = hp.Float("learning_rate", min_value=1e-7, max_value=1e-2, sampling="log")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=smape, 
        metrics=['mae', 'mse', 'mape', smape]
        )

    return model


def get_tuner(model, directory, window, overwrite=False):
    for example_inputs, example_labels in window.train.take(1):
        out_steps = example_labels.shape[1]
        number_of_features = example_labels.shape[-1]
    
    if model == 'ARF':
        hypermodel = lambda hp: build_ARF(hp, out_steps, number_of_features)
    elif model == 'SS':
        hypermodel = lambda hp: build_SS(hp, out_steps, number_of_features)
    elif model == 'ED':
        hypermodel = lambda hp: build_ED(hp, out_steps, number_of_features)
    else:
        raise NotImplementedError('This model type is not known.')
    
    tuner = keras_tuner.Hyperband(
        hypermodel=hypermodel,
        objective=keras_tuner.Objective("val_smape", direction="min"),
        max_epochs=200,
        factor=3,
        hyperband_iterations=1,
        executions_per_trial=3,
        seed=2023,
        max_retries_per_trial=10,
        max_consecutive_failed_trials=10,
        overwrite=overwrite,
        directory=directory,
        project_name=model,
    )
    return tuner

def run(tuner, window):
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    tuner.search(window.train, validation_data=window.val, callbacks=[early_stopping_callback], verbose=2)
    

def final_train(tuner, window, checkpoint_path):
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor='val_smape', verbose=0, save_best_only=True, save_weights_only=True)
    
    best_hps = tuner.get_best_hyperparameters()[0]
    m = tuner.hypermodel.build(best_hps)
    m.fit(window.train, epochs=200, validation_data=window.val, callbacks=[checkpoint, early_stopping_callback], verbose=2)
    return m
    