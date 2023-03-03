import pandas as pd
import datetime
import numpy as np
import openpyxl
#import data
path_file_1_p = "/Users/martijnvroon/Documents/uni/master/seminar (master)/data/macro_economic_indicators/Population_projections.xlsx"
population_data = pd.read_excel(path_file_1_p)

#delete the last 5 empty rows 
num_rows = len(population_data)
population_data = population_data.drop(population_data.index[num_rows-5:num_rows])
#rename a column 
population_data = population_data.rename(columns = {'Country Name': 'PARTNER_Labels','Time':'TIME_PERIOD'})
population_data = population_data.rename(columns = {'Population, total [SP.POP.TOTL]': 'A'})

#get's the index s of the columns with string
df = population_data
string_indices = df['A'][df['A'].apply(lambda x: isinstance(x, str))].index


#gets th countries corresponging to the index s 
lll = len(string_indices)
LIST_of_del_countries = [None] * lll

counter = 0 
for i in range(lll):
    LIST_of_del_countries[counter] = population_data.iloc[(string_indices[i]),0]
    counter = counter +1

list1 = list(set(LIST_of_del_countries))

print(df.shape)
print(list1)
for j in range(4000):
  
    if df.iloc[j,0] in list1:
        df = df.drop(index=[j])
        print(df.shape)

df = df.replace('..', 0)
print(df)
# boolean indexing to select rows to delete
idx = df['PARTNER_Labels'].str.contains('IDA blend')

# drop rows using boolean indexing
df = df.drop(df[idx].index)

idx = df['PARTNER_Labels'].str.contains('IDA only')
df = df.drop(df[idx].index)

idx = df['PARTNER_Labels'].str.contains('IDA total')
df = df.drop(df[idx].index)

for name in list1:
    idx = df['PARTNER_Labels'].str.contains(name)
    df = df.drop(df[idx].index)


testing= df.copy()
excel_test = pd.ExcelWriter("population_data.xlsx")
testing.to_excel(excel_test)
excel_test.save()
#make the 
monthly_data_p = population_data.iloc[0:0].copy()

number_of_col = df.shape[1]
nan_row = np.empty((number_of_col,))
nan_row[:] = np.nan

n_row = df.shape[0] 
for i in range(n_row):
    print(i)
    holder_for_rows = df.iloc[i,:]
    current_country = df.iloc[i,0]
    current_country_code = df.iloc[i,1] 
    for j in range(12):
        index_3 = 12 * i + j
        if j == 0 :
            monthly_data_p.loc[index_3] = holder_for_rows   
        else:
            monthly_data_p.loc[index_3] = nan_row
            monthly_data_p.iloc[index_3,0] = current_country
            monthly_data_p.iloc[index_3,1] = current_country_code 


# testing= monthly_data_p.copy()
# excel_test = pd.ExcelWriter("dopper2.xlsx")
# testing.to_excel(excel_test)
# excel_test.save()

c_p = population_data.columns[4]
monthly_data_p[c_p] = monthly_data_p[c_p].apply(pd.to_numeric)
n_countries = len(df['Country Code'].unique())
#fills in the date column
for i in range(n_countries-10):
    index_1 = 0   + 228*i
    index_2 = 228 + 228*i
    monthly_data_p.iloc[index_1:index_2,4] = monthly_data_p.iloc[index_1:index_2,4].interpolate().copy()
    #adding dates 
    monthly_data_p.iloc[index_1:index_2,2] = pd.period_range("2005-01", "2023-12", freq='M')

# testing= monthly_data_p.copy()
# excel_test = pd.ExcelWriter("dopper3.xlsx")
# testing.to_excel(excel_test)
# excel_test.save()
