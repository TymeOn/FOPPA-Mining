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
def calculate_statistics():
    for file_name in file_names:
        if file_name == "Lots":
            # Select only numeric columns
            numeric_columns = ["lotId", "tedCanId", "correctionsNb", "awardEstimatedPrice", "awardPrice",
                               "numberTenders", "lotsNumber", "numberTendersSme", "contractDuration",
                               "publicityDuration"]
            numeric_data = dataframes[file_name][numeric_columns]

            # Calculate statistics
            stats = numeric_data.describe()

            # Calculate mode
            mode = numeric_data.mode().iloc[0]

            # Display statistics
            print(f"\nStatistiques pour le fichier {file_name}:")
            print(stats)

            # Display mode
            print("\n##### Mode #####")
            for col, val in mode.items():
                print(f"{col}: {val}")


# Main function
if __name__ == "__main__":

    # Loading the files into dataframes
    load_data()

    # DEBUG DISPLAY
    for filename, df in dataframes.items():
        print(f"DataFrame pour le fichier {filename}:")
        print(df.head())
        print("\n")

    # Calculating standard statistics
    calculate_statistics()

    # for file_name in file_names:
    #     dataframes[file_name].to_csv(file_name + "_clean.csv", index = False)

    # Mesure d'association (Corréaltion pisson standard)

