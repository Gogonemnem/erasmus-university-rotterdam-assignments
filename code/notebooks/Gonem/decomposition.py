import statsmodels.api as sm
import pandas as pd


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
    

class Filter():
    def __init__(self, labels=None):
        self.labels = labels

    def __call__(self, data):
        return data[self.labels]
    
def main():
    import openpyxl
    import pathlib
    cwd = pathlib.Path.cwd()

    # code_directory = cwd.parents[1]
    code_directory = cwd / "code"

    bas_directory = code_directory / "notebooks" / "Bas"
    data_file = bas_directory / "df_filtered_maize_trade_oil_weather_futures.xlsx"
    data_file


    df = pd.read_excel(data_file, header=[0, 1], index_col=0)
    label_columns = ['price']
    label_columns = df.columns[df.columns.get_level_values(0).isin(label_columns)].tolist()

    dc = STLDecomposer(labels=label_columns)
    print(dc.labels)
    print(dc(df).head(5))
    # print(decomposables.head(5))
    # print(untouched_data.head(5))

if __name__ == '__main__':
    main()