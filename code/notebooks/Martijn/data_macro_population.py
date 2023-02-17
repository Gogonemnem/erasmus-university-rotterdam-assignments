import pandas as pd
#import datetime
import numpy
import openpyxl

path_file_1_p = "/Users/martijnvroon/Documents/uni/master/seminar (master)/data/macro_economic_indicators/Population_projections.xlsx"
population_data = pd.read_excel(path_file_1_p)
filler_data_frame = population_data[0:0].copy()


# 266 * 19 = 5054
for i in range(266):
    index_1_p = 0  + 19*i
    index_2_p = 19 + 19*i
    
    df_p_dummy = population_data[index_1_p:index_2_p].copy()
    df_p_dummy.reset_index(drop=True, inplace=True)
    filler_data_frame = pd.concat([filler_data_frame, df_p_dummy ], axis= 1)

#remove the ducplication of afhganistan 
filler_data_frame = filler_data_frame.iloc[:, 5:]
#remove the year 2023
filler_data_frame = filler_data_frame[:-1]
#remove columns that are no time serie data 
filler_data_frame.drop([ 'Country Name', 'Country Code', 'Time',   'Time Code' ], axis= 1, inplace= True )


#convert to monthly data
monthly_data_p = filler_data_frame.iloc[0:0].copy()
number_of_col_p = monthly_data_p.shape[1]
nan_row = numpy.empty((number_of_col_p,))
nan_row[:] = numpy.nan
counter3 = 0
for p in range(18):
    counter3 = counter3 + 1
    holder_for_rows = filler_data_frame.iloc[p,:]
    for q in range(12):
        index_3_p = 12 * p + q
        if q == 0 :
            monthly_data_p.loc[index_3_p] = holder_for_rows 
        else:
            monthly_data_p.loc[index_3_p] = nan_row 


#reset column names to indexen 


number_of_col_p = monthly_data_p.shape[1]  
list1_p = range(number_of_col_p)
monthly_data_p.set_axis(list1_p, axis='columns', inplace=True)

#rename the columns naar country code
list_country_codes = population_data['Country Code'].unique()
list_country_codes= list_country_codes[:-1]

number_of_col_p = monthly_data_p.shape[1]  
list1_p = range(number_of_col_p)
monthly_data_p.set_axis(list_country_codes, axis='columns', inplace=True)

#makes sure that all the columns are in the right data type for the interpolate function
for col in monthly_data_p:
    monthly_data_p[col] = pd.to_numeric(monthly_data_p[col], errors='coerce')
#fill in the missing data 
filled = monthly_data_p.interpolate().copy()


pop_data= filled.copy()
excel_file_3_p = pd.ExcelWriter("populatie.xlsx")
pop_data.to_excel(excel_file_3_p)
excel_file_3_p.save()

#### in voegen van de andere data ############################################################################################



path_file_1 = "/Users/martijnvroon/Documents/uni/master/seminar (master)/DATA_AH_case_studies_ah/macro_economic_indicators/macro_economic_indicators.csv"

path_file_1 = "/Users/martijnvroon/Documents/uni/master/seminar (master)/data/macro_economic_indicators/macro_economic_indicators.csv"
economic_indicators = pd.read_csv(path_file_1, sep = ";", decimal=".")

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

#adding one year 2022
for y in range(10):
    zero_row = pd.DataFrame([[0]*monthly_data.shape[1]],columns=monthly_data.columns)
    monthly_data = monthly_data.append(zero_row, ignore_index=True)

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

#print(inter_data.shape)



#test area 
##############################################################################################################################
# print(population_data)
# print(population_data.columns)
# print(population_data.columns[1])

list_country_codes = population_data['Country Code'].unique()
list_country_codes= list_country_codes[:-1]




# print(economic_indicators.columns)

# print(economic_indicators.columns[1])
# print(economic_indicators['country_code'].unique())
list_macro_codes = economic_indicators['country_code'].unique()[:-1]
# print(list_macro_codes[0])
# print(list_country_codes[:-1])
# print(len(list_country_codes))
# print(len(list1_p))

number_of_col_p = monthly_data_p.shape[1]  
list1_p = range(number_of_col_p)
monthly_data_p.set_axis(list_country_codes, axis='columns', inplace=True)


test_test = inter_data.copy()

counter4 = 0
for c in list_macro_codes:
    #print(c)
    index_col = 0 + counter4 * 9
    counter4 = counter4 +1
    if c not in filled:
        print(c)
    if index_col > 1952:
        break
    dummy_for_country = filled[c]
    title_string = str(c)
    test_test.insert(loc = index_col, column = title_string, value = dummy_for_country)

testing= test_test.copy()
excel_test = pd.ExcelWriter("test.xlsx")
testing.to_excel(excel_test)
excel_test.save()

#print(monthly_data_p)
#test area 