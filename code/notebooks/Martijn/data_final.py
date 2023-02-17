import pandas as pd
#import datetime
import numpy
import openpyxl
#import data
path_file_1_p = "/Users/martijnvroon/Documents/uni/master/seminar (master)/data/macro_economic_indicators/Population_projections.xlsx"
population_data = pd.read_excel(path_file_1_p)

path_file_1 = "/Users/martijnvroon/Documents/uni/master/seminar (master)/data/macro_economic_indicators/macro_economic_indicators.csv"
economic_indicators = pd.read_csv(path_file_1, sep = ";", decimal=".")

#rename column for merge later
population_data = population_data.rename(columns = {'Country Name': 'PARTNER_Labels'})
economic_indicators = economic_indicators.rename(columns = {'country_name': 'PARTNER_Labels' ,'Time':'TIME_PERIOD' })
#this is a quick fix
macro_economic_indicators = economic_indicators
# Convert year to datetime with year and month
macro_economic_indicators['year'] = pd.to_datetime(macro_economic_indicators['year'], format='%Y').dt.to_period('M')
# Set the date as the index
macro_economic_indicators.set_index('year', inplace=True)

#make df for monthly data
monthly_data = macro_economic_indicators.iloc[0:0].copy()


number_of_col = macro_economic_indicators.shape[1]
nan_row = numpy.empty((number_of_col,))
nan_row[:] = numpy.nan

#add na rows for 11 months in a year
n_row = macro_economic_indicators.shape[0] 
for i in range(n_row):
    print(i)
    holder_for_rows = macro_economic_indicators.iloc[i,:]
    current_country = macro_economic_indicators.iloc[i,0] 
    current_country_code = macro_economic_indicators.iloc[i,1] 
    for j in range(12):
        index_3 = 12 * i + j
        if j == 0 :
            monthly_data.loc[index_3] = holder_for_rows   
        else:
            monthly_data.loc[index_3] = nan_row
            monthly_data.iloc[index_3,0] = current_country
            monthly_data.iloc[index_3,1] = current_country_code

#slects als the column with numbers
c = monthly_data.columns[3:12]
monthly_data[c] = monthly_data[c].apply(pd.to_numeric)

n_countries = len(population_data['Country Code'].unique())
#fills in the date column
for i in range(n_countries-1):
    index_1 = 0   + 204*i
    index_2 = 204 + 204*i
    monthly_data.iloc[index_1:index_2,:] = monthly_data.iloc[index_1:index_2,:].interpolate().copy()
    #adding dates 
    monthly_data.iloc[index_1:index_2,2] = pd.period_range("2005-01", "2021-12", freq='M')
     
#exports the df to excel
testing= monthly_data.copy()
excel_test = pd.ExcelWriter("bas.xlsx")
testing.to_excel(excel_test)
excel_test.save()



