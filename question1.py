import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import seaborn as sns

dataframe = pd.read_csv("./learningData/question1.csv", sep=';', decimal=',', low_memory=False)
dataframe = dataframe.dropna(subset=['topType'])
dataframe = dataframe[dataframe['awardPrice'] > 0]
dataframe = dataframe[dataframe['distance'] > 0]
feature_names = ['awardPrice', 'distance']
topType = dataframe['topType'].unique()

print("> Chargement des données:")
average_awardPrice = {t: 0 for t in topType}
average_distance = {t: 0 for t in topType}
frequencys = {t: 0 for t in topType}
for i, row in dataframe.iterrows():
    average_awardPrice[row['topType']] += row['awardPrice']
    average_distance[row['topType']] += row['distance']
    frequencys[row['topType']] += 1
    # Calculer les moyennes et les fréquences pour chaque topType
        
for t in topType:
    average_awardPrice[t] = average_awardPrice[t]/frequencys[t]
    average_distance[t] = average_distance[t]/frequencys[t]

print(frequencys)
print("> Frequence:")
df = pd.DataFrame(list(frequencys.items()), columns=['topType', 'Frequency'])
fig = plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='topType', y='Frequency', order=df['topType'], palette='viridis')
plt.title('Frequency of Each Top Type')
plt.xlabel('Top Type')
plt.ylabel('Frequency')
plt.savefig('./figure_question1/frequence.png')
plt.close(fig)

print("> Correlation:")
# Normaliser la fréquence pour utiliser comme diamètre
frequency = list(frequencys.values())
frequency = np.array(frequency)
frequency = (frequency - frequency.min()) / (frequency.max() - frequency.min())

average_distance = list(average_distance.values())
average_awardPrice = list(average_awardPrice.values())
# Créer le graphique de dispersion
fig = plt.figure(figsize=(10, 6))
plt.scatter(average_distance, average_awardPrice, s=5000*frequency, alpha=0.5)

# Annoter chaque point avec le topType correspondant
for i, txt in enumerate(topType):
    plt.annotate(txt, (average_distance[i], average_awardPrice[i]), textcoords="offset points", xytext=(0,10), ha='center')


# Ajouter des titres et des étiquettes
plt.title('Corrélation entre awardPrice et distance par topType')
plt.xlabel('Distance')
plt.ylabel('Award Price')
plt.grid(True)
plt.savefig('./figure_question1/correlation')
plt.close(fig)


print("> Cluster:")
#########################################################################
# 3 - Cluster

# Convert string labels to numerical values
label_encoder = LabelEncoder()
dataframe['topType'] = label_encoder.fit_transform(dataframe['topType'])
X = dataframe[feature_names]
y = dataframe['topType']

# Normalize data
scaler = MinMaxScaler()
X_norm = scaler.fit_transform(X)

# Plot the ground truth with logarithmically scaled data
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)

for label in dataframe['topType'].unique():
    ax.scatter(X.loc[dataframe['topType'] == label, 'awardPrice'], 
               X.loc[dataframe['topType'] == label, 'distance'], 
               edgecolor='k', label=topType[label])

ax.legend()
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.set_xlabel('awardPrice')
ax.set_ylabel('distance')
ax.set_title('Répartition des données')

plt.grid(True)
plt.savefig('./figure_question1/k-means')
plt.close(fig)