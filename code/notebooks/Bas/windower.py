
import tensorflow as tf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import openpyxl
import pathlib
cwd = pathlib.Path.cwd()

code_directory = cwd.parents[1]
# code_directory = cwd / "code"

bas_directory = code_directory / "notebooks" / "Bas"
data_file = bas_directory / "df_filtered_maize_trade_oil_weather_futures.xlsx"
data_file

df = pd.read_excel(data_file, header=[0, 1], index_col=0)

class WindowGenerator():
    def __init__(self, input_width, label_width, shift, data=df,
                 split=[0, 0.7, 0.9, 1], label_columns=None):
        
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.fillna(0)
        
        n = len(data)
        self.split = split
        self.train_df = data[int(n*split[0]):int(n*split[1])]
        self.val_df   = data[int(n*split[1]):int(n*split[2])]
        self.test_df  = data[int(n*split[2]):int(n*split[3])]

        # Store the raw data.
        self.raw_train_df = self.train_df
        self.raw_val_df   = self.val_df
        self.raw_test_df  = self.test_df

        self.train_mean = self.train_df.mean()
        self.train_std  = self.train_df.std().replace(0, 1)

        # self.train_df = (self.train_df - self.train_mean) / self.train_std
        # self.val_df   = (self.val_df   - self.train_mean) / self.train_std
        # self.test_df  = (self.test_df  - self.train_mean) / self.train_std

        self.label_columns = label_columns
        self._set_indices()

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result
    
    def set_label_columns(self, label_columns):
        self.label_columns = label_columns
        self._set_indices()
            
    def _set_indices(self):
        # Work out the label column indices.
        if self.label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(self.label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(self.train_df.columns)}
            
    def preprocess(self, preprocessor):
        self.train_df = preprocessor(self.train_df)
        self.val_df = preprocessor(self.val_df)
        self.test_df = preprocessor(self.test_df)
        self._set_indices()

    def copy(self, df=None):
        # if df is None:
        df = pd.concat([self.train_df, self.val_df, self.test_df])

        new_window = WindowGenerator(input_width=self.input_width, 
                                     label_width=self.label_width, 
                                     shift=self.shift, 
                                     data=df,
                                     split=self.split,
                                     label_columns=self.label_columns)
        
        return new_window

        
    # This function splits the whole window into the input data X and the labels y
    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels
    
    def plot(self, model=None, plot_col=None, max_subplots=3):
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n+1)
            plt.ylabel(f'{plot_col}')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                    label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(inputs)

                print(label_col_index)
                print(predictions.shape)
                
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                # plt.scatter(self.label_indices, predictions[n, label_col_index],
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Time [h]')
        plt.show()

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            ###### STILL PLAYING WITH SHUFFLE
            shuffle=True,
            batch_size=32,)

        ds = ds.map(self.split_window)

        return ds

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

def main():
    label_columns = ['price']
    all_label_columns = df.columns[df.columns.get_level_values(0).isin(label_columns)].tolist()

    w1 = WindowGenerator(input_width=24, label_width=1, shift=24, 
                        train_df=train_df, val_df=val_df, 
                        test_df=test_df, label_columns=all_label_columns)

    w1


    w2 = WindowGenerator(input_width=6, label_width=1, shift=1,
                        train_df=train_df, val_df=val_df, 
                        test_df=test_df, label_columns=all_label_columns)
    w2


    # Stack three slices, the length of the total window.
    example_window = tf.stack([np.array(train_df[:w2.total_window_size]),
                            np.array(train_df[50:50+w2.total_window_size]),
                            np.array(train_df[100:100+w2.total_window_size])])

    example_inputs, example_labels = w2.split_window(example_window)

    print('All shapes are: (batch, time, features)')
    print(f'Window shape: {example_window.shape}')
    print(f'Inputs shape: {example_inputs.shape}')
    print(f'Labels shape: {example_labels.shape}')


    # w1.plot(plot_col=all_label_columns[2])
    # all_label_columns[1]


    # Each element is an (inputs, label) pair.
    w2.train.take(32)


    for example_inputs, example_labels in w2.train.take(1):
        print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
        print(f'Labels shape (batch, time, features): {example_labels.shape}')


    # Extract the indices of the highest level 'Level1'
    level1_indices = df.columns.get_level_values(0).isin(['price'])

    # Use boolean indexing to get the numerical indices of the desired level
    level1_indices = np.where(level1_indices)[0]
    level1_indices



if __name__ == "__main__":
    main()