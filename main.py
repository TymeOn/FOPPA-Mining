import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

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

dataframes_all = {}

column_types_all = {
    "Lots": {
        "lotId": "Int64",
        "tedCanId": "Int64",
        "correctionsNb": "Int64",
        "cancelled": "bool",
        "awardDate": "str",
        "awardEstimatedPrice": "Float64",
        "awardPrice": "Float64",
        "cpv": "str",
        "numberTenders": "Int64",
        "onBehalf": "bool",
        "jointProcurement": "bool",
        "fraAgreement": "bool",
        "fraEstimated": "str",
        "lotsNumber": "str",
        "accelerated": "bool",
        "outOfDirectives": "bool",
        "contractorSme": "bool",
        "numberTendersSme": "Int64",
        "subContracted": "bool",
        "gpa": "bool",
        "multipleCae": "str",
        "typeOfContract": "str",
        "topType": "str",
        "renewal": "bool",
        "contractDuration": "Float64",
        "publicityDuration": "Float64",
        "suppliersId": "Int64",
        "buyersId": "Int64",
        "criterionId": "Int64",
        "name_criteria": "str",
        "weight": "Float64",
        "type": "str",
        "name_suppliers": "str",
        "siret_suppliers": "str",
        "address_suppliers": "str",
        "city_suppliers": "str",
        "zipcode_suppliers": "str",
        "country_suppliers": "str",
        "department_suppliers": "str",
        "longitude_suppliers": "Float64",
        "latitude_suppliers": "Float64",
        "name_buyersId": "str",
        "siret_buyersId": "str",
        "address_buyersId": "str",
        "city_buyersId": "str",
        "zipcode_buyersId": "str",
        "country_buyersId": "str",
        "department_buyersId": "str",
        "longitude_buyersId": "Float64",
        "latitude_buyersId": "Float64",
        "name2_buyers": "str",
        "name2_suppliers": "str",
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
    # Create directory if it doesn't exist
    directory = "removedData/"
    if not os.path.exists(directory):
        os.makedirs(directory)

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
            if not essentials_columns[file_name]:
                na_free = dataframe.dropna()
            else:
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


def merge_csv():
    data_lot_suppliers = dataframes["LotSuppliers"].copy()
    data_lot_suppliers.rename(columns={"agentId": "suppliersId"}, inplace=True)

    data_lot_buyers = dataframes["LotBuyers"].copy()
    data_lot_buyers.rename(columns={"agentId": "buyersId"}, inplace=True)

    data_lots = dataframes["Lots"].copy()

    data_lots = pd.merge(data_lots, data_lot_suppliers, on="lotId", how='left')
    data_lots = pd.merge(data_lots, data_lot_buyers, on="lotId", how='left')

    data_criteria = dataframes["Criteria"].copy()
    data_criteria.rename(columns={"name": "name_criteria"}, inplace=True)
    data_lots = pd.merge(data_lots, data_criteria, on="lotId", how='left')

    data_agents_suppliers = dataframes["Agents"].copy()
    data_agents_suppliers.rename(columns={"agentId": "suppliersId"}, inplace=True)
    data_agents_suppliers.rename(columns={"name": "name_suppliers"}, inplace=True)
    data_agents_suppliers.rename(columns={"siret": "siret_suppliers"}, inplace=True)
    data_agents_suppliers.rename(columns={"address": "address_suppliers"}, inplace=True)
    data_agents_suppliers.rename(columns={"city": "city_suppliers"}, inplace=True)
    data_agents_suppliers.rename(columns={"zipcode": "zipcode_suppliers"}, inplace=True)
    data_agents_suppliers.rename(columns={"country": "country_suppliers"}, inplace=True)
    data_agents_suppliers.rename(columns={"department": "department_suppliers"}, inplace=True)
    data_agents_suppliers.rename(columns={"longitude": "longitude_suppliers"}, inplace=True)
    data_agents_suppliers.rename(columns={"latitude": "latitude_suppliers"}, inplace=True)
    data_lots = pd.merge(data_lots, data_agents_suppliers, on="suppliersId", how='left')

    data_agents_buyers = dataframes["Agents"].copy()
    data_agents_buyers.rename(columns={"agentId": "buyersId"}, inplace=True)
    data_agents_buyers.rename(columns={"name": "name_buyers"}, inplace=True)
    data_agents_buyers.rename(columns={"siret": "siret_buyers"}, inplace=True)
    data_agents_buyers.rename(columns={"address": "address_buyers"}, inplace=True)
    data_agents_buyers.rename(columns={"city": "city_buyers"}, inplace=True)
    data_agents_buyers.rename(columns={"zipcode": "zipcode_buyers"}, inplace=True)
    data_agents_buyers.rename(columns={"country": "country_buyers"}, inplace=True)
    data_agents_buyers.rename(columns={"department": "department_buyers"}, inplace=True)
    data_agents_buyers.rename(columns={"longitude": "longitude_buyers"}, inplace=True)
    data_agents_buyers.rename(columns={"latitude": "latitude_buyers"}, inplace=True)

    data_lots = pd.merge(data_lots, data_agents_buyers, on="buyersId", how='left')

    data_names_suppliers = dataframes["Names"].copy()
    data_names_suppliers.rename(columns={"agentId": "suppliersId"}, inplace=True)
    data_names_suppliers.rename(columns={"name": "name2_suppliers"}, inplace=True)

    names_suppliers_dict = data_names_suppliers.set_index('suppliersId').to_dict()
    data_lots['name2_suppliers'] = data_lots['suppliersId'].map(names_suppliers_dict['name2_suppliers'])

    data_names_buyers = dataframes["Names"].copy()
    data_names_buyers.rename(columns={"agentId": "buyersId"}, inplace=True)
    data_names_buyers.rename(columns={"name": "name2_buyers"}, inplace=True)

    names_buyers_dict = data_names_buyers.set_index('buyersId').to_dict()
    data_lots['name2_buyers'] = data_lots['buyersId'].map(names_buyers_dict['name2_buyers'])

    data_lots.to_csv("data/merged_lots_data.csv", index=False)


# Loading the specified CSV files into dataframes
def load_data_all():
    global dataframes_all
    path = "data/merged_lots_data.csv"

    # Loading the individual file into a dataframe
    if os.path.exists(path):
        print("> Chargement du fichier  merged_lots_data ...")
        dataframes_all = pd.read_csv(path, dtype=column_types_all["Lots"])
        dataframes_all[integer_columns] = dataframes_all[integer_columns].apply(pd.to_numeric, errors='coerce')
    else:
        print(f"Le fichier {path} n'existe pas.")
    return


# Tracer le cercle de corrélation
def correlation_circle(components, var_names, x_axis, y_axis, status):
    fig, axes = plt.subplots(figsize=(8, 8))
    minx = -1
    maxx = 1
    miny = -1
    maxy = 1
    axes.set_xlim(minx, maxx)
    axes.set_ylim(miny, maxy)
    # label with variable names
    # ignore first variable (instance name)
    for i in range(0, components.shape[1]):
        axes.arrow(0,
                   0,  # Start the arrow at the origin
                   components[i, x_axis],  # 0 for PC1
                   components[i, y_axis],  # 1 for PC2
                   head_width=0.01,
                   head_length=0.02)

        plt.text(components[i, x_axis] + 0.05,
                 components[i, y_axis] + 0.05,
                 var_names[i])
    # axes
    plt.plot([minx, maxx], [0, 0], color='silver', linestyle='-', linewidth=1)
    plt.plot([0, 0], [miny, maxy], color='silver', linestyle='-', linewidth=1)
    # add a circle
    cercle = plt.Circle((0, 0), 1, color='blue', fill=False)
    axes.add_artist(cercle)
    plt.savefig('fig/acp_correlation_circle_axes_' + status + str(x_axis) + '_' + str(y_axis))
    plt.close(fig)


def question2_analytics_bar(dataframes_dt_suppliers, dataframes_dt_buyers):
    dataframes_dt_suppliers.plot(kind='bar', stacked=True, figsize=(22, 6))
    plt.title('Nombre de lots par département fournisseur et par type de contrat')
    plt.xlabel('Département fournisseur')
    plt.ylabel('Nombre de lots')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('fig/lots_par_departement_contrat_suppliers.png')

    dataframes_dt_buyers.plot(kind='bar', stacked=True, figsize=(22, 6))
    plt.title('Nombre de lots par département acheteur et par type de contrat')
    plt.xlabel('Département fournisseur')
    plt.ylabel('Nombre de lots')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('fig/lots_par_departement_contrat_buyers.png')

    plt.figure(figsize=(22, 8))
    dataframes_all['department_suppliers'].value_counts().plot(kind='bar', color='skyblue')
    plt.title('Nombre de lots par département fournisseur')
    plt.xlabel('Département fournisseur')
    plt.ylabel('Nombre de lots')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('fig/nombre_lots_par_departement_suppliers.png')

    plt.figure(figsize=(22, 8))
    dataframes_all['department_buyers'].value_counts().plot(kind='bar', color='skyblue')
    plt.title('Nombre de lots par département acheteur')
    plt.xlabel('Département fournisseur')
    plt.ylabel('Nombre de lots')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('fig/nombre_lots_par_departement_buyers.png')


def question2_analytics_hierarchical(dataframes_dt_suppliers, dataframes_dt_buyers):
    # Sélection des départements comme étiquettes
    lst_labels_suppliers = dataframes_dt_suppliers.index
    lst_labels_buyers = dataframes_dt_buyers.index

    # Construction de la matrice de distances
    Z_suppliers = linkage(dataframes_dt_suppliers.values, method='ward')
    Z_buyers = linkage(dataframes_dt_buyers.values, method='ward')

    # Tracé du dendrogramme
    plt.figure(figsize=(12, 6))
    dendrogram(Z_suppliers, labels=lst_labels_suppliers, orientation='top', leaf_rotation=90)
    plt.title('Dendrogramme de clustering hiérarchique des départements fournisseurs')
    plt.xlabel('Départements fournisseurs')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.savefig('fig/dendrogram_clustering_hierarchical_suppliers.png')

    # Tracé du dendrogramme
    plt.figure(figsize=(12, 6))
    dendrogram(Z_buyers, labels=lst_labels_buyers, orientation='top', leaf_rotation=90)
    plt.title('Dendrogramme de clustering hiérarchique des départements acheteurs')
    plt.xlabel('Départements acheteurs')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.savefig('fig/dendrogram_clustering_hierarchical_buyers.png')


def question2_analytics_pca(dataframes_dt_suppliers, dataframes_dt_buyers):
    # Réaliser une PCA sur les données
    pca_suppliers = PCA()
    pca_buyers = PCA()

    pca_suppliers.fit(dataframes_dt_suppliers[['S', 'U', 'W']])
    pca_buyers.fit(dataframes_dt_buyers[['S', 'U', 'W']])

    # Calculer les composantes principales (PC)
    components_suppliers = pca_suppliers.components_.T * np.sqrt(pca_suppliers.explained_variance_)
    components_buyers = pca_buyers.components_.T * np.sqrt(pca_buyers.explained_variance_)

    # Noms des variables
    var_names = ['S', 'U', 'W']

    correlation_circle(components_suppliers, var_names, 0, 1, 'suppliers')
    correlation_circle(components_buyers, var_names, 0, 1, 'buyers')

    # Sélectionner uniquement les caractéristiques à utiliser pour le clustering
    X_suppliers = dataframes_dt_suppliers[['S', 'U', 'W']]
    X_buyers = dataframes_dt_buyers[['S', 'U', 'W']]

    # Normaliser les données
    scaler_suppliers = MinMaxScaler()
    scaler_buyers = MinMaxScaler()

    X_norm_suppliers = scaler_suppliers.fit_transform(X_suppliers)
    X_norm_buyers = scaler_buyers.fit_transform(X_buyers)

    # Effectuer le clustering KMeans
    kmeans_suppliers = KMeans(n_clusters=3, random_state=42)
    kmeans_buyers = KMeans(n_clusters=3, random_state=42)

    kmeans_suppliers.fit(X_norm_suppliers)
    kmeans_buyers.fit(X_norm_buyers)

    labels_suppliers = kmeans_suppliers.labels_
    labels_buyers = kmeans_buyers.labels_

    # Afficher les clusters
    plt.figure(figsize=(22, 6))
    plt.scatter(dataframes_dt_suppliers.index, labels_suppliers, c=labels_suppliers, cmap='viridis')
    plt.title('Clustering KMeans des départements fournisseurs')
    plt.xlabel('Département fournisseur')
    plt.ylabel('Cluster')
    plt.xticks(rotation=45)
    plt.savefig('fig/kmeans_department_clusters_suppliers')

    # Afficher les clusters
    plt.figure(figsize=(22, 6))
    plt.scatter(dataframes_dt_buyers.index, labels_buyers, c=labels_buyers, cmap='viridis')
    plt.title('Clustering KMeans des départements acheteurs')
    plt.xlabel('Département acheteur')
    plt.ylabel('Cluster')
    plt.xticks(rotation=45)
    plt.savefig('fig/kmeans_department_clusters_buyers')


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

    # Utilisation de la fonction merge_csv
    merge_csv()

    # Loading data merged into dataframes all
    load_data_all()

    dataframes_dt_suppliers = pd.pivot_table(dataframes_all, index='department_suppliers', columns='typeOfContract',
                                             aggfunc='size', fill_value=0)

    print(dataframes_dt_suppliers)

    dataframes_dt_buyers = pd.pivot_table(dataframes_all, index='department_buyers', columns='typeOfContract',
                                          aggfunc='size', fill_value=0)

    question2_analytics_bar(dataframes_dt_suppliers, dataframes_dt_buyers)
    question2_analytics_hierarchical(dataframes_dt_suppliers, dataframes_dt_buyers)
    question2_analytics_pca(dataframes_dt_suppliers, dataframes_dt_buyers)

    # for file_name in file_names:
    #     dataframes[file_name].to_csv(file_name + "_clean.csv", index = False)
