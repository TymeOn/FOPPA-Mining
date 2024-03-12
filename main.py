import pandas as pd
import os

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

            dataframes[file_name] = dataframe
        else:
            print(f"Le fichier {path} n'existe pas.")

    return


# Calculating standard statistics
def calculate_standard_statistics(table_name):
    print(f"\nStatistiques standard pour le fichier {table_name} :")

    for column, dtype in column_types[table_name].items():
        if dtype == "Int64" or dtype == "Float64":
            print(f"Champ: {column}")
            numeric_data = dataframes[table_name][column]

            # Calculate statistics
            stats = numeric_data.describe()

            # Display statistics
            print(stats)


def str_statistics(table_name):
        str_data = dataframes[table_name]

        # Parcourir chaque colonne
        for column, dtype in column_types[table_name].items():
            if dtype == "str":
                unique_values = str_data[column].unique()

                # Si le nombre de valeurs uniques est supérieur à 1, il y a des valeurs différentes
                if len(unique_values) > 1:
                    print(f"\nChamp: {column}")

                    # Afficher la valeur la plus courante
                    most_common_value = str_data[column].mode().iloc[0]
                    print(f"\nLa valeur la plus courante : {most_common_value}")

                    print(f"\nValeurs différentes : {unique_values}")

                    # Afficher le nombre de fois que chaque valeur apparaît
                    value_counts = str_data[column].value_counts()
                    print(f"\nNombre de fois que chaque valeur apparaît : {value_counts}")

                    # Afficher le nombre de valeurs uniques
                    unique_count = len(unique_values)
                    print(f"\nNombre de valeurs uniques dans la colonne '{column}' : {unique_count}")


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
    print(f"Nombre de fois que chaque combinaison de latitude et longitude apparaît dans le dataframe Agents :")
    print(coordinates.value_counts())

    # Afficher la valeur la plus courante
    most_common_coordinates = coordinates.mode().iloc[0]
    print(f"La combinaison de latitude et longitude la plus courante dans le dataframe Agents : {most_common_coordinates}")

    # Afficher les valeurs différentes
    unique_coordinates = coordinates.unique()
    print(f"Combinaisons de latitude et longitude différentes dans le dataframe Agents : {unique_coordinates}")

    # Afficher le nombre total de combinaisons uniques de latitude et longitude
    unique_count = len(unique_coordinates)
    print(f"Nombre total de combinaisons uniques de latitude et longitude dans le dataframe Agents : {unique_count}")


def display_criterion_count_per_lot():
    """
    Affiche le nombre de criterionId par lotId.
    """
    dataframeCriteria = dataframes['Criteria']

    # Regrouper les données par lotId et compter le nombre de criterionId uniques dans chaque groupe
    criterion_count_per_lot = dataframeCriteria.groupby("lotId")["criterionId"].nunique()

    print(f"Nombre de criterionId par lotId dans le dataframe Criteria :")
    print(criterion_count_per_lot)

    print(f"Statistiques descriptives du nombre de criterionId par lotId dans le dataframe Criteria :")
    print(criterion_count_per_lot.describe())


def describe_weight_of_criteria():
    """
    Affiche les statistiques descriptives du poids (weight) de Criteria,
    ainsi que le mode.
    """
    criteriaDataframe = dataframes["Criteria"].dropna(subset=['weight'])

    print("Statistiques descriptives du poids (weight) de Criteria:")
    print(criteriaDataframe["weight"].describe())

    # Afficher le mode du poids
    mode_weight = criteriaDataframe["weight"].mode()
    print("Mode du poids (weight) de Criteria:")
    print(mode_weight)


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
    describe_weight_of_criteria()

    calculate_standard_statistics("Names")
    str_statistics("Names")

    # for file_name in file_names:
    #     dataframes[file_name].to_csv(file_name + "_clean.csv", index = False)

    # Mesure d'association (Corréaltion pisson standard)

