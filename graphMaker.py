import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import csv

# fabrique les graphiques de fréquence
def graph_frequency_maker(data, column):
    data = data.dropna().sort_values()
    chart_data = {}
    max_value = int(max(data))
    min_value = int(min(data))
    print(column, ":\n")
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
    # Créer un graphique
    x_values = list(chart_data.keys())
    y_values = list(chart_data.values())
    delimiter = ';'
    csv_file_path = f"dataProduce/{column}.csv"
    with open(csv_file_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=delimiter)

        csv_writer.writerow(['Number', 'Frequency'])

        for x, y in zip(x_values, y_values):
            csv_writer.writerow([x, y])
    
    plt.figure(figsize=(10, 6))
    plt.scatter(x_values, y_values, s=50, color='blue', alpha=0.7)
    plt.xlabel('Number')
    plt.ylabel('Frequency')
    plt.title(f"Frequency Chart {column}")
    plt.xlim(min(x_values), max(x_values))
    plt.savefig(f"charts_frequency/{column}_frequency_chart.png")
    
def graph_double_maker(data, column1, column2):
    # creer les catégories uniques
    print(f"Graphiques {column1} et {column2}")
    categories = [category for category in data[column1] if category is not None and category != 'NaN']
    unique_categories = set(categories)
    
    # initialiser sommes et fréquences
    print("categorie de la classe: ", unique_categories)
    sommes = {category: 0 for category in unique_categories}
    frequency = {category: 0 for category in unique_categories}

    # itérer le data frame
    for i, row in data.iterrows():
        if row[column1] in unique_categories and row[column2] is not None and isinstance(row[column2], (int, float)):
            sommes[row[column1]] += float(row[column2])
            frequency[row[column1]] += 1

    # faire les moyennes
    averages = {key: sommes[key] / frequency[key] if frequency[key] != 0 else 0 for key in unique_categories}
    
    # trier les frequences et les sommes
    frequency = dict(sorted(frequency.items()))
    sommes = dict(sorted(sommes.items()))
    averages = dict(sorted(averages.items()))
    
    print("moyennes : ", averages)
    print("fréquence : ", frequency)
    print("sommes : ", sommes)
    #Create a frequency histogram
    plt.figure(figsize=(10, 6))
    plt.bar(frequency.keys(), frequency.values(), color='red', alpha=0.7)
    plt.xlabel(column1)
    plt.ylabel(f'Frequency {column2}')
    plt.title(f"Frequency {column2} by {column1}")
    plt.savefig(f"charts_two_columns/frequency_{column1}_{column2}_chart.png")
    
    #Create a somme histogram
    plt.figure(figsize=(10, 6))
    plt.bar(sommes.keys(), sommes.values(), color='green', alpha=0.7)
    plt.xlabel(column1)
    plt.ylabel(f'Sommes {column2}')
    plt.title(f"Sommes {column2} by {column1}")
    plt.savefig(f"charts_two_columns/sommes_{column1}_{column2}_chart.png")
    
    #Create a average histogram
    plt.figure(figsize=(10, 6))
    plt.bar(averages.keys(), averages.values(), color='blue', alpha=0.7)
    plt.xlabel(column1)
    plt.ylabel(f'Average {column2}')
    plt.title(f"Average {column2} by {column1}")
    plt.savefig(f"charts_two_columns/average_{column1}_{column2}_chart.png")
    print("\n")

def tableau_boolean(Lots, boolean_columns, number_columns):
    # Filter the DataFrame based on boolean columns
    Lots[boolean_columns] = Lots[boolean_columns].astype(int)
    Lots["bool_id"] = Lots[boolean_columns].astype(str).agg(''.join, axis=1)
    unique_bools = set(Lots["bool_id"])
    unique_bools = list(unique_bools)

    sommes_dict = {
        'bool_id': unique_bools,
    }

    frequency_dict = {
        'bool_id': unique_bools,
    }

    average_dict = {
        'bool_id': unique_bools,
    }

    for col in boolean_columns:
        sommes_dict[col] = [0] * len(unique_bools)
        frequency_dict[col] = [0] * len(unique_bools)
        average_dict[col] = [0] * len(unique_bools)

    for col in number_columns:
        sommes_dict[col] = [0] * len(unique_bools)
        frequency_dict[col] = [0] * len(unique_bools)
        average_dict[col] = [0] * len(unique_bools)
    
    # Processing the data
    for column in number_columns:
        print("> traitement colonne: ", column)
        for i, row in Lots.iterrows():
            if row["bool_id"] in unique_bools and row[column] is not None and isinstance(row[column], (int, float)):
                bool_id = row['bool_id']
                
                # Update sommes_dict
                idx = sommes_dict['bool_id'].index(bool_id)
                sommes_dict[column][idx] += row[column]
                
                # Update frequency_dict
                idx = frequency_dict['bool_id'].index(bool_id)
                frequency_dict[column][idx] += 1

    
    # Calculate average
    for column in number_columns:
        for idx, bool_val in enumerate(unique_bools):
            if frequency_dict[column][idx] > 0:
                sommes_dict[column][idx] = int(sommes_dict[column][idx])
                average_dict[column][idx] = round(sommes_dict[column][idx] / frequency_dict[column][idx], 2)
    
    for index, col in enumerate(boolean_columns):
        for idx, key in enumerate(unique_bools):
            sommes_dict[col][idx] = int(key[index])
            frequency_dict[col][idx] = int(key[index])
            average_dict[col][idx] = int(key[index])
            
    del sommes_dict['bool_id']
    del frequency_dict['bool_id']
    del average_dict['bool_id']
    
    print(average_dict)
    # Convert dictionaries to DataFrames
    sommes = pd.DataFrame(sommes_dict)
    frequency = pd.DataFrame(frequency_dict)
    average = pd.DataFrame(average_dict)

    # Save DataFrames as CSV files
    sommes.to_csv("boolean_matrices/sommes.csv", sep=';', decimal=',', index=False)
    frequency.to_csv("boolean_matrices/frequency.csv", sep=';', decimal=',', index=False)
    average.to_csv("boolean_matrices/average.csv", sep=';', decimal=',', index=False)
