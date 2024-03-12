import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import csv
import graphMaker

dataframes = {}
file_names = ["Agents", "Criteria", "LotBuyers", "Lots", "LotSuppliers", "Names"]
column_types = {
    "Agents": {
        "agentId": "Int64",
        "name": "str",
        "siret": "str",
        "address": "str",
        "city": "str",
        "zipcode": "str",
        "country": "str",
        "department": "str",
        "longitude": "Float64",
        "latitude": "Float64"
    },
    "Criteria": {
        "criterionId": "Int64",
        "lotId": "Int64",
        "name": "str",
        "weight": "Float64",
        "type": "str"
    },
    "LotBuyers": {
        "lotId": "Int64",
        "agentId": "Int64"
    },
    "Lots": {
        "lotId": "Int64",
        "tedCanId": "Int64",
        "correctionsNb": "Int64",
        "cancelled": "str", # bool
        "awardDate": "str",
        "awardEstimatedPrice": "Float64",
        "awardPrice": "Float64",
        "cpv": "str",
        "numberTenders": "Int64",
        "onBehalf": "str", # bool
        "jointProcurement": "str", # bool
        "fraAgreement": "str", # bool
        "fraEstimated": "str",
        "lotsNumber": "str", # Int64
        "accelerated": "str", # bool
        "outOfDirectives": "str", # bool
        "contractorSme": "str", # bool
        "numberTendersSme": "Int64",
        "subContracted": "str", # bool
        "gpa": "str", # bool
        "multipleCae": "str",
        "typeOfContract": "str",
        "topType": "str",
        "renewal": "str", # bool
        "contractDuration": "Float64",
        "publicityDuration": "Float64"
    },
    "LotSuppliers": {
        "lotId": "Int64",
        "agentId": "Int64"
    },
    "Names": {
        "agentId": "Int64",
        "names": "str"
    }
}
boolean_columns = [
    "cancelled",
    "onBehalf",
    "jointProcurement",
    "fraAgreement",
    "accelerated",
    "outOfDirectives",
    "contractorSme",
    "subContracted",
    "gpa",
    "renewal"
]
integer_columns = [
    "lotsNumber"
]
limit_columns = {
    "awardEstimatedPrice": 1000000,
    "awardPrice": 1000000,
    "contractDuration": 365,
    "lotsNumber": 300,
    "publicityDuration": 120,
    "numberTenders": 120
}
limit_counter = {}
essentials_columns = {
    "Agents" : ["agentId", "name", "siret", "address", "city", "zipcode", "country"],
    "Criteria" : [],
    "LotBuyers" : [],
    "Lots" : ["lotId", "lotsNumber", "awardPrice", "awardDate", "typeOfContract", "numberTenders"],
    "LotSuppliers" : [],
    "Names" : [],
}

# Loading the specified CSV files into dataframes
def load_data():
    for file_name in file_names:
        path = "data/" + file_name + ".csv"

        # Loading the individual file into a dataframe
        if os.path.exists(path):
            print("> Chargement du fichier " + file_name + "...")
            dataframe = pd.read_csv(path, dtype=column_types[file_name])

            if file_name == "Lots":
                dataframe[boolean_columns] = dataframe[boolean_columns].map(lambda x: True if x == "Y" else False)
                dataframe[integer_columns] = dataframe[integer_columns].apply(pd.to_numeric, errors='coerce')
                for col, limit in limit_columns.items():
                    initial_na = dataframe[col].isna().sum()
                    dataframe[col] = dataframe[col].map(lambda x: x if x is None or pd.isna(x) or (0 <= x <= limit) else None)
                    final_na = dataframe[col].isna().sum()
                    limit_counter[col] = final_na - initial_na


            if essentials_columns[file_name] == [] :
                na_free = dataframe.dropna()
            else :
                na_free = dataframe.dropna(subset=essentials_columns[file_name])
            only_na = dataframe[~dataframe.index.isin(na_free.index)]
            only_na.to_csv("removedData/removed" + file_name + ".csv", sep=';', decimal=',', float_format='%.3f')
            dataframe = na_free

            dataframes[file_name] = dataframe
        else:
            print(f"Le fichier {path} n'existe pas.")

    return

number_lots_columns = [
    "awardEstimatedPrice",
    "awardPrice",
    "numberTenders",
    "lotsNumber",
    "numberTendersSme",
    "contractDuration",
    "publicityDuration"
]


    
# Schéma et donnnées 
def data_analysis():
    for column in number_lots_columns:
        plt.title(f"{column} frequence")
        data = pd.to_numeric(dataframes["Lots"][column], errors='coerce')
        graphMaker.graph_frequency_maker(data, column)
    data = pd.to_numeric(dataframes["Criteria"]["weight"], errors='coerce')
    graphMaker.graph_frequency_maker(data, column)
    
    data1 = dataframes["Lots"]["typeOfContract"]
    for number_lots_column in number_lots_columns:
        data2 = pd.to_numeric(dataframes["Lots"][number_lots_column], errors='coerce')

        # Create a new DataFrame with the desired columns
        data = pd.DataFrame({'typeOfContract': data1, number_lots_column: data2})
        
        graphMaker.graph_double_maker(data, "typeOfContract", number_lots_column)
    
    data1 = dataframes["Lots"]["awardDate"].str[:4]
    for number_lots_column in number_lots_columns:
        data2 = pd.to_numeric(dataframes["Lots"][number_lots_column], errors='coerce')

        # Create a new DataFrame with the desired columns
        data = pd.DataFrame({'awardDate': data1, number_lots_column: data2})
        
        graphMaker.graph_double_maker(data, "awardDate", number_lots_column)


#
# Main function
if __name__ == "__main__":

    # Loading the files into dataframes
    load_data()

    # DEBUG DISPLAY
    
    for filename, df in dataframes.items():
        print(f"DataFrame pour le fichier {filename}:")
        print(df.head())
        print("\n")

    data_analysis()
    # for file_name in file_names:
    #     dataframes[file_name].to_csv(file_name + "_clean.csv", index = False)

