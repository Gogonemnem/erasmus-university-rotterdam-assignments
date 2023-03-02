
import tensorflow as tf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import openpyxl
import pathlib
cwd = pathlib.Path.cwd()

code_directory = cwd.parents[1]
# code_directory = cwd / "code"

# bas_directory = code_directory / "notebooks" / "Bas"
# data_file = bas_directory / "df_filtered_maize_trade_oil_weather_futures.xlsx"
# data_file

# df = pd.read_excel(data_file, header=[0, 1], index_col=0)

class WindowGenerator():
    def __init__(self, input_width, label_width, shift, data, train_begin=0, train_end=.7, val_begin=None, val_end=None,
                 test_begin=None, test_end=None, connect=False, remove_labels=False, label_columns=None):
        self.connect = connect
        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.total_window_size = input_width + shift
        
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.fillna(0)

        self.train_val_test_split(data, train_begin, train_end, val_begin, val_end, test_begin, test_end)

        # Store the raw data.
        self.raw_train_df = self.train_df
        self.raw_val_df   = self.val_df
        self.raw_test_df  = self.test_df

        self.train_mean = self.train_df.mean()
        self.train_std  = self.train_df.std().replace(0, 1)

        # self.train_df = (self.train_df - self.train_mean) / self.train_std
        # self.val_df   = (self.val_df   - self.train_mean) / self.train_std
        # self.test_df  = (self.test_df  - self.train_mean) / self.train_std

        self.remove_labels = remove_labels
        self.label_columns = label_columns
        self._set_indices()

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    @property
    def train(self):
        result = self.make_dataset(self.train_df)
        if self.remove_labels:
            self._train_labeled = result
            result = self.remove_label_columns(result)
        self._train = result
        return result

    @property
    def val(self):
        result = self.make_dataset(self.val_df)
        if self.remove_labels:
            self._val_labeled = result
            result = self.remove_label_columns(result)
        self._val = result
        return result

    @property
    def test(self):
        result = self.make_dataset(self.test_df)
        if self.remove_labels:
            self._test_labeled = result
            result = self.remove_label_columns(result)
        self._test = result
        return result

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            if self.remove_labels:
                result = next(iter(self._train_labeled))
            else: 
                result = next(iter(self._train))
            # And cache it for next time
            self._example = result
        return result

    def train_val_test_split(self, data, train_begin=0, train_end=.7, val_begin=None, val_end=None, test_begin=None, test_end=None):
        n = len(data)

        if not isinstance(train_begin, int):
            train_begin = int(n*train_begin)
        if not isinstance(train_end, int):
            train_end = int(n*train_end)

        if val_begin is None:
            if val_end is None:
                val_begin, val_end = train_begin, train_end
            elif self.connect:
                val_begin = train_end - self.total_window_size + 1
            else:
                val_begin = train_end
        elif not isinstance(val_begin, int):
            val_begin = int(n*val_begin)
        
        if val_end is None:
            if test_begin is None:
                val_end = n - 0.5*(n - val_begin)
            elif self.connect:
                val_end = test_begin + self.total_window_size - 1
            else:
                val_end = test_begin
        elif not isinstance(val_end, int):
            val_end = int(n*val_end)
    
        if test_begin is None:
            if test_end is None:
                test_begin, test_end = val_begin, val_end
            elif self.connect:
                test_begin = val_end - self.total_window_size + 1
            else:
                test_begin = val_end
        elif not isinstance(test_begin, int):
            test_begin = int(n*test_begin)

        if test_end is None:
            test_end = n
        elif not isinstance(test_end, int):
            test_end = int(n*test_end)      

        self.train_df = data[train_begin:train_end]
        self.val_df   = data[val_begin:val_end]
        self.test_df  = data[test_begin:test_end]

    def remove_label_columns(self, dataset):
        if self.label_columns is None:
            return dataset
        
        column_indices = [self.train_df.columns.get_loc(idx) for idx in self.label_columns]
        column_indices = np.setdiff1d(np.arange(self.train_df.shape[1]), column_indices)
        return dataset.map(lambda x, y: (tf.gather(x, column_indices, axis=2), y))
    
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

    def add_label_postprocess(self, label_postprocessor):
        self.label_postprocessor = label_postprocessor

    def copy(self, df=None):
        # if df is None:
        df = pd.concat([self.train_df, self.val_df, self.test_df])

        new_window = WindowGenerator(input_width=self.input_width, 
                                     label_width=self.label_width, 
                                     shift=self.shift, 
                                     data=df,
                                     split=self.split,
                                     label_columns=self.label_columns,
                                     remove_labels=self.remove_labels)
        
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
        if self.remove_labels:
            inputs_full, labels = self.example
            # column_indices = [self.train_df.columns.get_loc(idx) for idx in self.label_columns]
            column_indices = [self.column_indices[name] for name in self.label_columns_indices]
            input_labels = tf.gather(inputs_full, column_indices, axis=2)
            column_indices = np.setdiff1d(np.arange(self.train_df.shape[1]), column_indices)
            inputs = tf.gather(inputs_full, column_indices, axis=2)
            

        else:
            inputs, labels = self.example
            inputs_full, labels_full = inputs, labels
            column_indices = [self.column_indices[name] for name in self.label_columns_indices]
            input_labels = tf.gather(inputs_full, column_indices, axis=2)

        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n+1)
            plt.ylabel(f'{plot_col}')
            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue
            # column_indices = [self.column_indices[name] for name in self.label_columns_indices]
            # inputs_labels = self.label_postprocessor.reverse(inputs_full[:, :, self.column_indices])
            # plt.plot(self.input_indices, inputs_full[n, :, plot_col_index],
            #         label='Inputs', marker='.', zorder=-10)
            plt.plot(self.input_indices, self.label_postprocessor.reverse(input_labels)[n, :, label_col_index],
                    label='Inputs', marker='.', zorder=-10)

            

            plt.scatter(self.label_indices, self.label_postprocessor.reverse(labels)[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(inputs)
                
                plt.scatter(self.label_indices, self.label_postprocessor.reverse(predictions)[n, :, label_col_index],
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
            shuffle=False,
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