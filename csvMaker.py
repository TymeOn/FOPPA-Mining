import numpy as np
import pandas as pd
from geopy.distance import geodesic

def calculate_distance(row):
    supplier_coords = (row['latitude_suppliers'], row['longitude_suppliers'])
    buyer_coords = (row['latitude_buyers'], row['longitude_buyers'])
    return geodesic(supplier_coords, buyer_coords).kilometers

def question1_csv(dataframe):
    dataframe1 = dataframe.dropna(subset=['longitude_suppliers', 'latitude_suppliers', 'longitude_buyers', 'latitude_buyers', 'typeOfContract',  'topType', 'awardPrice'])
    dataframe1['distance'] = dataframe1.apply(calculate_distance, axis=1)
    result = dataframe1[['typeOfContract', 'topType', 'awardPrice', 'distance']]
    result.to_csv("learningData/question1.csv", sep=';', decimal=',', index=False)
    
def question2_csv(dataframe):
    dataframe1 = dataframe.dropna(subset=['numberTenders', 'awardPrice', 'department_suppliers', 'department_buyers'])
    result = dataframe1[['numberTenders', 'awardPrice', 'department_suppliers', 'department_buyers']]
    result.to_csv("learningData/question2.csv", sep=';', decimal=',', index=False)
    
def question2_type_csv(dataframe):
    dataframe1 = dataframe.dropna(subset=['numberTenders', 'awardPrice', 'department_suppliers', 'department_buyers', 'typeOfContract'])
    dataframe1 = dataframe1.sort_values(by='typeOfContract')
    types_of_contracts = dataframe1['typeOfContract'].unique()
    
    for contract_type in types_of_contracts:
        subset_df = dataframe1[dataframe1['typeOfContract'] == contract_type]
        result = subset_df[['numberTenders', 'awardPrice', 'department_suppliers', 'department_buyers']]
        result.to_csv(f"learningData/question2_{contract_type}.csv", sep=';', decimal=',', index=False)

if __name__ == "__main__":
    dataframe = pd.read_csv("./data/merged_lots_data.csv", low_memory=False)
    question1_csv(dataframe)
    question2_csv(dataframe)
    question2_type_csv(dataframe)