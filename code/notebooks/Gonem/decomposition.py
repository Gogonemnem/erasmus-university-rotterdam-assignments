import statsmodels.api as sm
import pandas as pd
from collections import deque
import math
import numpy as np

class Processor():
    def __init__(self):
        self.processors = []
    
    def add(self, processor):
        self.processors.append(processor)
        return self
    
    def __call__(self, data):
        for processor in self.processors:
            data = processor(data)
        return data
    
    def reverse(self, data):
        for processor in self.processors:
            data = processor.reverse(data)
        return data

class STLDecomposer():
    def __init__(self, labels=None, period=None, smoothing_param=7):
        ## Assumption that the data will be a pandas dataframe with column labels
        self.labels = labels
        self.period = period
        self.smoothing_param = smoothing_param

    def set_labels(self, labels):
        self.labels = labels
    
    def _split_decomposables(self, data):
        decomposables = data[self.labels]
        untouched_data = data.drop(columns=self.labels)
        return decomposables, untouched_data

    def __call__(self, data):
        decomposables, untouched_data = self._split_decomposables(data)

        def get_stl_results(column):
            stl = sm.tsa.STL(column, period=self.period, seasonal=self.smoothing_param).fit()
            return pd.DataFrame({(column.name[0]+'_trend', column.name[1]): stl.trend, 
                                 (column.name[0]+'_seasonal', column.name[1]): stl.seasonal, 
                                 (column.name[0]+'_residual', column.name[1]): stl.resid})
        
        stl_components = pd.concat([get_stl_results(decomposables[col]) for col in decomposables.columns], axis=1)
        combined_data = pd.concat([stl_components, untouched_data, decomposables], axis=1)
        return combined_data
    
    def reverse(self, data):
        return data
    

class Standardizer():
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        if self.mean is None:
            self.mean = data.mean()
            self.std  = data.std()
            # avoid infinity
            self.std = self.std.replace(0, 1)

        data = (data - self.mean) / self.std
        return data
    
    def reverse(self, data):
        data = (data * self.std) + self.mean
        return data
    

class Logger():
    def __init__(self, factor=math.e, labels=None, label_indices=None) -> None:
        self.factor = factor
        self.labels = labels
        self.label_indices = label_indices # data.columns.get_indexer(self.labels)

    def __call__(self, data):
        data[self.labels] = np.log(data[self.labels] + 1e-8) / np.log(self.factor)
        return data
    
    def reverse(self, data):
        if isinstance(data, pd.DataFrame):
            data[self.labels] = self.factor ** data[self.labels]
        else: # data is of form (batch, time, features)
            for i in self.label_indices:
                data_i = data[:, :, i]
                data_i = self.factor ** (data_i + 1e-8)
                data_i = np.expand_dims(data_i, axis=-1)
                data = np.concatenate([data[:, :, :i], data_i, data[:, :, i+1:]], axis=-1)
        return data
class Filter():
    def __init__(self, labels=None):
        self.labels = labels

    def __call__(self, data):
        if self.labels is not None:
            data = data[self.labels]
        return data
    
    def reverse(self, data):
        if self.labels is not None:
            data = data.drop(self.labels, axis=1)
        return data
    
def main():
    import openpyxl
    import pathlib
    cwd = pathlib.Path.cwd()

    # code_directory = cwd.parents[1]
    code_directory = cwd / "code"

    gonem_directory = code_directory / "notebooks" / "Gonem"
    data_file = gonem_directory / "MAIZE_FILTERED_2023-02-25_19-36-41.xlsx"
    data_file

    df = pd.read_excel(data_file, header=[0, 1], index_col=0)

    label_columns = ['price']
    label_columns = df.columns[df.columns.get_level_values(0).isin(label_columns)].tolist()
    
    preproc = Processor()
    
    stl = STLDecomposer(labels=label_columns, period=12)
    std = Standardizer()
    log = Logger(labels=label_columns)

    preproc.add(stl).add(log).add(std)
    # print(df['price'].describe())
    # df = preproc(df)

    # df = preproc.reverse(df)
    # print(df['price'].describe())

    from windower import WindowGenerator

    width = 12
    label_width = 3
    shift = 3

    

    w = WindowGenerator(input_width=width, label_width=label_width, shift=shift, data=df, 
                        split=[0, 0.6, 0.85, 1], remove_labels=True, label_columns=label_columns)
    w.preprocess(preproc)

    
    label_std = Standardizer(mean=std.mean[label_columns], std=std.std[label_columns])
    label_log = Logger(label_indices=range(len(label_columns)))
    postproc = Processor().add(label_std).add(label_log)
    for x, y in w.train.take(1):
        print(y)
        # print(label_std.reverse(y))
        print(postproc.reverse(y))

    
    

    # preproc(df)



    # label_columns = ['price']
    # label_columns = df.columns[df.columns.get_level_values(0).isin(label_columns)].tolist()

    # dc = STLDecomposer(labels=label_columns)
    # print(dc.labels)
    # print(dc(df).head(5))
    # print(decomposables.head(5))
    # print(untouched_data.head(5))

if __name__ == '__main__':
    main()