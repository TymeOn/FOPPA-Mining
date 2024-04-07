import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import seaborn as sns
import sys

dataframe = pd.read_csv(f"./learningData/{sys.argv[1]}.csv", sep=';', decimal=',', low_memory=False)
dataframe = dataframe.dropna(subset=[sys.argv[2]])
dataframe = dataframe[dataframe[sys.argv[3]] > 0]
dataframe = dataframe[dataframe[sys.argv[4]] > 0]
feature_names = [sys.argv[3], sys.argv[4]]
type = dataframe[sys.argv[2]].unique()

print("> Chargement des données:")
average1 = {t: 0 for t in type}
average2 = {t: 0 for t in type}
frequencys = {t: 0 for t in type}
for i, row in dataframe.iterrows():
    average1[row[sys.argv[2]]] += row[sys.argv[3]]
    average2[row[sys.argv[2]]] += row[sys.argv[4]]
    frequencys[row[sys.argv[2]]] += 1
    # Calculer les moyennes et les fréquences pour chaque topType
        
for t in type:
    average1[t] = average1[t]/frequencys[t]
    average2[t] = average2[t]/frequencys[t]

print(frequencys)
print("> Frequence:")
df = pd.DataFrame(list(frequencys.items()), columns=['topType', 'Frequency'])
fig = plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='topType', y='Frequency', order=df['topType'], palette='viridis')
plt.title('Freaquence pour chaque type')
plt.xlabel('Top Type')
plt.ylabel('Frequency')
plt.savefig(f"./figure_{sys.argv[1]}/frequence_{sys.argv[2]}.png")
plt.close(fig)

print("> Correlation:")
# Normaliser la fréquence pour utiliser comme diamètre
frequency = list(frequencys.values())
frequency = np.array(frequency)
frequency = (frequency - frequency.min()) / (frequency.max() - frequency.min())

average2 = list(average2.values())
average1 = list(average1.values())
# Créer le graphique de dispersion
fig = plt.figure(figsize=(10, 6))
plt.scatter(average2, average1, s=5000*frequency, alpha=0.5)

# Annoter chaque point avec le topType correspondant
for i, txt in enumerate(type):
    plt.annotate(txt, (average2[i], average1[i]), textcoords="offset points", xytext=(0,10), ha='center')


# Ajouter des titres et des étiquettes
plt.title(f"Corrélation entre {sys.argv[3]} et {sys.argv[4]} par {sys.argv[2]}")
plt.xlabel(sys.argv[4])
plt.ylabel(sys.argv[3])
plt.grid(True)
plt.savefig(f"./figure_{sys.argv[1]}/correlation_{sys.argv[2]}_{sys.argv[3]}_{sys.argv[4]}")
plt.close(fig)


print("> Cluster:")
#########################################################################
# 3 - Cluster

# Convert string labels to numerical values
label_encoder = LabelEncoder()
dataframe[sys.argv[2]] = label_encoder.fit_transform(dataframe[sys.argv[2]])
X = dataframe[feature_names]
y = dataframe[sys.argv[2]]

# Normalize data
scaler = MinMaxScaler()
X_norm = scaler.fit_transform(X)

# Plot the ground truth with logarithmically scaled data
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)

for label in dataframe[sys.argv[2]].unique():
    ax.scatter(X.loc[dataframe[sys.argv[2]] == label, sys.argv[3]], 
               X.loc[dataframe[sys.argv[2]] == label, sys.argv[4]], 
               edgecolor='k', label=type[label])

ax.legend()
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.set_xlabel(sys.argv[3])
ax.set_ylabel(sys.argv[4])
ax.set_title('Répartition des données')

plt.grid(True)
plt.savefig(f"./figure_{sys.argv[1]}/k-means_{sys.argv[2]}_{sys.argv[3]}_{sys.argv[4]}")
plt.close(fig)