import pandas as pd
import openpyxl # Needed for reading excel
import pathlib


def get_data(file_path=None, directory_path=None, product=None):
    if file_path is None:
        file_path = get_file_path(product, directory_path)
    
    return pd.read_excel(file_path, header=[0, 1], index_col=0)
    
        
            
def get_file_path(product, path=None):
    if product == 'maize':
        file_name = "MAIZE_FILTERED_2023-03-03_02-09-43.xlsx"
    elif product == 'sunflower':
        file_name = "SUNFLOWER_FILTERED_2023-03-03_02-19-29.xlsx"
    elif product == 'wheat':
        file_name = "WHEAT_FILTERED_2023-03-03_02-44-24.xlsx"
    else:
        raise NotImplementedError("This product is not known to us")

    if path is None:
        return file_name
    return path / file_name