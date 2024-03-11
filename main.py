import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import csv

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

                na_free = dataframe.dropna(subset=["lotsNumber", "awardPrice", "awardDate", "typeOfContract", "numberTenders"])
                only_na = dataframe[~dataframe.index.isin(na_free.index)]
                only_na.to_csv("removedData/removed" + file_name + ".csv", sep=';', decimal=',', float_format='%.3f')
                dataset = na_free

            dataframes[file_name] = dataframe
        else:
            print(f"Le fichier {path} n'existe pas.")

    return

number_lots_columns = [
    # "correctionsNb",
    "awardEstimatedPrice",
    "awardPrice",
    "numberTenders",
    "lotsNumber",
    "numberTendersSme",
    "contractDuration",
    "publicityDuration"
]

def data_analysis():
    for number_lots_column in number_lots_columns:
        plt.title(f"{number_lots_column} frequence")
        data = pd.to_numeric(dataframes["Lots"][number_lots_column], errors='coerce')
        data = data.dropna().sort_values()
        chart_data = {}
        max_value = int(max(data))
        min_value = int(min(data))
        print(number_lots_column, ":\n")
        print("max: ", max_value, "\n")
        print("min: ", min_value, "\n")
        
        # Partager le tableau de valeurs du graphique en part égales
        nb_part = 50
        step = int((max_value - min_value)/nb_part)
        if(step < 1):step =1
        index = int(min_value)
        while index < max_value:
            chart_data[index] = 0
            if index < 0:
                chart_data[0] = 0
            index += step
        # Ajoute les fréquences des termes dans le tableau
        for value in data:
            # trouve la bonne catégorie pour la valeur
            for i in chart_data.keys():
                if int(value) < int(i):
                    break
                index = i
            chart_data[index] += 1
            
        print(chart_data)
        # Créer un histogramme
        x_values = list(chart_data.keys())
        y_values = list(chart_data.values())
        delimiter = ';'
        csv_file_path = f"troubles/{number_lots_column}.csv"
        with open(csv_file_path, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=delimiter)

            csv_writer.writerow(['Number', 'Frequency'])

            for x, y in zip(x_values, y_values):
                csv_writer.writerow([x, y])
        
        plt.figure(figsize=(10, 6))
        plt.scatter(x_values, y_values, s=50, color='blue', alpha=0.7)
        plt.xlabel('Number')
        plt.ylabel('Frequency')
        plt.title(f"Frequency Chart {number_lots_column}")
        plt.xlim(min(x_values), max(x_values))
        plt.savefig(f"charts/{number_lots_column}_frequency_chart.png")

# Main function
if __name__ == "__main__":

    # Loading the files into dataframes
    load_data()

    # DEBUG DISPLAY

    print("Valeurs numériques dépassant les limites définies :", limit_counter)
    
    for filename, df in dataframes.items():
        print(f"DataFrame pour le fichier {filename}:")
        print(df.head())
        print("\n")

    data_analysis()
    # for file_name in file_names:
    #     dataframes[file_name].to_csv(file_name + "_clean.csv", index = False)

