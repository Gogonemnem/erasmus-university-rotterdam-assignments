import pandas as pd
import numpy
import openpyxl
import datetime
import pathlib

cwd = pathlib.Path.cwd()
code_directory = cwd / "code"
data_directory = code_directory / "data"
trade_data_directory = data_directory / "trade_data"
macro_data_directory = data_directory / "macro_economic_indicators"

# path_file_1 = "/Users/martijnvroon/Documents/uni/master/seminar (master)/DATA_AH_case_studies_ah/macro_economic_indicators/macro_economic_indicators.csv"
economic_indicators = pd.read_csv(macro_data_directory / "macro_economic_indicators.csv", sep = ";", decimal=".")

df = economic_indicators[0:17].copy()
df.reset_index(drop=True, inplace=True)
df.drop([ 'gdp_current_us','country_name','country_code', 'year', 'timecode'    , 'gdp_constant_2015_us' ,'gdp_constant_2015_us',\
        'population_total'  ,'unemployment_total',  'agricultural_land', 'renewable_energy_consumption_perc_of_total' ,\
        'energy_use_kg_of_oil_equivalent_per_capita' ,'fossil_fuel_energy_consumption_perc_of_total'], axis= 1, inplace= True )


# this loop rearranges the data in oder of time instead of country 
for i in range(217):
    index_1 = 0  + 17*i 
    index_2 = 17 + 17*i

    df_dummy = economic_indicators[index_1:index_2].copy()
    df_dummy.reset_index(drop=True, inplace=True)
    #let op ,'country_name'
    df_dummy.drop(  ['country_code','country_name', 'year', 'timecode'], axis= 1, inplace= True )
    df = pd.concat([df, df_dummy ], axis= 1)

#replaces the colum names with indexes   1953
number_of_col = df.shape[1]  
list1 = range(number_of_col)
df.set_axis(list1, axis='columns', inplace=True)


#save to a excel file
#df_new = df.copy()
#excel_file = pd.ExcelWriter("back_up_values.xlsx")
#df_new.to_excel(excel_file)
#excel_file.save()

#quick convert to montly
monthly_data = df.iloc[0:0].copy()

number_of_col = monthly_data.shape[1]

nan_row = numpy.empty((number_of_col,))
nan_row[:] = numpy.nan

counter3 = 0
for p in range(17):
    counter3 = counter3 + 1
    holder_for_rows = df.iloc[p,:]
    for q in range(12):
        index_3 = 12 * p + q
        if q == 0 :
            monthly_data.loc[index_3] = holder_for_rows 
        else:
            monthly_data.loc[index_3] = nan_row 


inter_data = monthly_data.interpolate().copy()
#print(monthly_data)

#monthly_data_new = monthly_data.copy()
#excel_file_2 = pd.ExcelWriter("monthly_data.xlsx")
#monthly_data_new.to_excel(excel_file_2)
#excel_file_2.save()

inter_data.fillna(0 ,inplace=True) 

#this saves the data to an excel file 
inter_data_new= inter_data.copy()
excel_file_3 = pd.ExcelWriter("inter_data.xlsx")
inter_data_new.to_excel(excel_file_3)
excel_file_3.save()


