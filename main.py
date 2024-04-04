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
        "name": "str"
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
    
    #tableau booléen
    graphMaker.tableau_boolean(dataframes["Lots"], boolean_columns, number_lots_columns)


# Calculating standard statistics
def calculate_standard_statistics(table_name):
    print(f"\nStatistiques standard pour le fichier {table_name} :")

    for column, dtype in column_types[table_name].items():
        if dtype == "Int64" or dtype == "Float64":
            print(f"Champ: {column}")
            numeric_data = dataframes[table_name][column]

            # Calculate statistics
            stats = numeric_data.describe()

            message = "Statistiques standard pour le champ " + column + " :"
            save_statistics(table_name, column, stats, "standard", message)

            # Display statistics
            print(stats)


def str_statistics(table_name):
        str_data = dataframes[table_name]

        # Parcourir chaque colonne
        for column, dtype in column_types[table_name].items():
            print(f"\nChamp: {column}")
            # Afficher la valeur la plus courante
            most_common_value = str_data[column].mode().iloc[0]
            message= "La valeur la plus courante : "
            print(f"\n{message}{most_common_value}")
            save_statistics(table_name, column, most_common_value, "str", message)

            if dtype == "str":
                unique_values = str_data[column].unique()

                # Si le nombre de valeurs uniques est supérieur à 1, il y a des valeurs différentes
                if len(unique_values) > 1:
                    message = "Valeurs différentes :"
                    print(f"\n{message} {unique_values}")
                    save_statistics(table_name, column, unique_values, "str", message)

                    # Afficher le nombre de fois que chaque valeur apparaît
                    value_counts = str_data[column].value_counts()
                    message = "Nombre de fois que chaque valeur apparaît : "
                    print(f"\n{message} {value_counts}")
                    save_statistics(table_name, column, value_counts, "str", message)

                    # Afficher le nombre de valeurs uniques
                    unique_count = len(unique_values)
                    message = "Nombre de valeurs uniques : "
                    print(f"\n{message} {unique_count}")
                    save_statistics(table_name, column, unique_count, "str", message)


def display_unique_coordinates():
    """
    Affiche le nombre total de combinaisons uniques de latitude et longitude,
    ainsi que le nombre de fois que chaque valeur apparaît, la valeur la plus courante,
    et les valeurs différentes.
    """

    dataframeAgent = dataframes['Agents'].dropna(subset=['latitude', 'longitude'])

    # Créer une colonne unique pour la position en concaténant latitude et longitude
    coordinates = dataframeAgent["latitude"].astype(str) + "," + dataframeAgent["longitude"].astype(str)

    # Afficher le nombre de fois que chaque valeur apparaît
    coordinates_count = coordinates.value_counts()
    message = "Nombre de fois que chaque combinaison de latitude et longitude apparaît"
    print(f"{message} dans le dataframe Agents : {coordinates_count}")
    save_statistics("Agents", "latitude-longitude", coordinates_count, "str", message + " : ")

    # Afficher la valeur la plus courante
    most_common_coordinates = coordinates.mode().iloc[0]
    message = "La combinaison de latitude et longitude la plus courante"
    print(f"{message} dans le dataframe Agents : {most_common_coordinates}")
    save_statistics("Agents", "latitude-longitude", most_common_coordinates, "str", message + " : ")

    # Afficher les valeurs différentes
    unique_coordinates = coordinates.unique()
    message = "Combinaisons de latitude et longitude différentes"
    print(f"{message} dans le dataframe Agents : {unique_coordinates}")
    save_statistics("Agents", "latitude-longitude", unique_coordinates, "str", message + " : ")

    # Afficher le nombre total de combinaisons uniques de latitude et longitude
    unique_count = len(unique_coordinates)
    message = "Nombre total de combinaisons uniques de latitude et longitude"
    print(f"{message} dans le dataframe Agents : {coordinates_count}")
    save_statistics("Agents", "latitude-longitude", coordinates_count, "str", message + " : ")


def display_criterion_count_per_lot():
    """
    Affiche le nombre de criterionId par lotId.
    """
    dataframeCriteria = dataframes['Criteria']

    # Regrouper les données par lotId et compter le nombre de criterionId uniques dans chaque groupe
    criterion_count_per_lot = dataframeCriteria.groupby("lotId")["criterionId"].nunique()
    message = "Nombre de criterionId par lotId"
    print(f"{message} dans le dataframe Criteria : {criterion_count_per_lot}")
    save_statistics("Criteria", "lotId-criterionId", criterion_count_per_lot, "str", message + " : ")

    message = "Statistiques descriptives du nombre de criterionId par lotId"
    print(f"{message} dans le dataframe Criteria : ")
    print(criterion_count_per_lot.describe())
    save_statistics("Criteria", "lotId-criterionId", criterion_count_per_lot.describe(), "standard", message + " : ")


def save_statistics(table_name, column, stats, status, message):
    # Create directory if it doesn't exist
    directory = "stats/" + table_name
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Write statistics to file
    file_path = os.path.join(directory, f"{column}_stats.txt")
    with open(file_path, "a") as file:
        if os.path.exists(file_path):
            if status == "str":
                if message == "Nombre de fois que chaque valeur apparaît : ":
                    file.write(f"{message}\n")
                    file.write(f"\n")
                    for index, value in stats.items():
                        file.write(f"• {index} : {value}\n")
                    file.write("\n")
                elif message == "Valeurs différentes :":
                    file.write(f"{message}\n")
                    file.write(f"\n")
                    for item in stats:
                        file.write(f"• {item}\n")
                    file.write("\n")
                else:
                    file.write(f"{message}{str(stats)}\n")
                    file.write(f"\n")
            if status == "standard":
                file.write(f"{message}\n")
                file.write(f"\n")
                file.write(stats.to_string() + "\n")
                file.write(f"\n")


# Main function
if __name__ == "__main__":

    # Loading the files into dataframes
    load_data()

    # DEBUG DISPLAY
    
    for filename, df in dataframes.items():
        print(f"DataFrame pour le fichier {filename}:")
        print(df.head())
        print("\n")

    # All statistics
    calculate_standard_statistics("Lots")
    str_statistics("Lots")

    calculate_standard_statistics("Agents")
    str_statistics("Agents")
    display_unique_coordinates()

    calculate_standard_statistics("Criteria")
    str_statistics("Criteria")
    display_criterion_count_per_lot()

    calculate_standard_statistics("Names")
    str_statistics("Names")

    data_analysis()
    
    # for file_name in file_names:
    #     dataframes[file_name].to_csv(file_name + "_clean.csv", index = False)
