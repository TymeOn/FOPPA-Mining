import pandas as pd
import os

dataframes = {}
file_names = ["Agents", "Criteria", "LotBuyers", "Lots", "LotSuppliers", "Names"]

# Loading the specified CSV files into dataframes
def load_data():
    for file_name in file_names:
        path = "data/" + file_name + ".csv"

        # Loading the individual file into a dataframe
        if os.path.exists(path):
            print("> Chargement du fichier " + file_name + "...")
            dataframe = pd.read_csv(path)
            dataframes[file_name] = dataframe
        else:
            print(f"Le fichier {path} n'existe pas.")

    return

# Main function
if __name__ == "__main__":

    # Loading the files into dataframes
    load_data()

    # DEBUG DISPLAY
    for filename, df in dataframes.items():
        print(f"DataFrame pour le fichier {filename}:")
        print(df.head())
        print("\n")