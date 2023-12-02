# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 21:06:17 2023

@author: danie
"""

#%% Importar librerias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn import preprocessing
#Configuración de las gráficas
plt.rcParams['figure.figsize'] = (13, 13)
plt.style.use('ggplot')

#%% Base de datos
data = pd.read_csv('segmentation data.csv')

data.info()

res = data.describe()

# Quitando columan ID
data2 = data.drop(columns=['ID'])

# Escalando los datos
scaler = preprocessing.StandardScaler()
scaler.fit(data2)
X = scaler.transform(data2)

#%% Clusterig jerarquico scipy
HC= hierarchy.linkage(X, metric='euclidean', method='complete')
#method:single,complete,avergae,centroid, ward,mean
#metric: minkowski, cityblock,euclidean,cosine,correlación,
#jaccard,chebyshev,hamming,mahalanobis,etc...
#note:centroid, median y ward sólo con Euclidean

#Dendograma
plt.title('Dendrograma Completo')
plt.xlabel('Índice de la muestra')
plt.ylabel('Distancia')
dn = hierarchy.dendrogram(HC)
plt.show()

#%% Modificar el aspecto del dendrograma
#plt.figure(figsize=(25,15))
plt.figure()
plt.title('Dendrograma Truncado')
plt.xlabel('índice de la muestra')
plt.ylabel('Distancia')
dn = hierarchy.dendrogram(HC, truncate_mode='level',p=3)
plt.show()

#%% Clustering jerárquico sklearn
clusters = 4
cluster = AgglomerativeClustering(n_clusters=clusters, affinity='euclidean', linkage='complete')  
grupos=cluster.fit_predict(X)

#%%Extraer los elementos de cada clúster

# Creando columna con edad e ingreso agrupados para mejorar la interpretación de los clusters
valores = [-999, 28, 38, 48, 58, 68, 999]
categorias = ['18-28', '28-38', '38-48', '48-58', '58-68', '68-76']
data['Age Bracket'] = pd.cut(x=data['Age'], bins=valores, labels=categorias)

valores = [-999, 81421, 127010, 172599, 218188, 263777, 99999999]
categorias = ['35,832-81,421', '81,421-127,010', '127,010-172,599', '172,599-218,188',\
              '218,188-263,777', '263,777-309,364']
data['Income Bracket'] = pd.cut(x=data['Income'], bins=valores, labels=categorias)

def analizar_cluster(X, data, cluster_id):
    """Analiza un cluster específico."""
    idx = grupos == cluster_id
    

    # Conteos
    usuario_counts = len(data[:][idx])
    sex_counts = pd.value_counts(data['Sex'][idx])
    marital_counts = pd.value_counts(data['Marital status'][idx])
    age_counts = pd.value_counts(data['Age Bracket'][idx])
    education_counts = pd.value_counts(data['Education'][idx])
    income_counts = pd.value_counts(data['Income Bracket'][idx])
    occupation_counts = pd.value_counts(data['Occupation'][idx])
    settlement_counts = pd.value_counts(data['Settlement size'][idx])

    return  usuario_counts, sex_counts, marital_counts, age_counts,\
        education_counts, income_counts, occupation_counts, settlement_counts

for i in range(clusters):
    usuario_counts, sex_counts, marital_counts, age_counts, education_counts,\
        income_counts, occupation_counts, settlement_counts = analizar_cluster(X, data, i)

    print(f"Cluster {i+1} - Análisis")
    print("--------------------------")
    print("\nConteo de Usuarios:")
    print(usuario_counts)
    print("\nConteo de Sex:")
    print(sex_counts)
    print("\nConteo de Marital:")
    print(marital_counts)
    print("\nConteo de Age:")
    print(age_counts)
    print("\nConteo de Education:")
    print(education_counts)
    print("\nConteo de Income:")
    print(income_counts)
    print("\nConteo de Occupation:")
    print(occupation_counts)
    print("\nConteo de Settlement:")
    print(settlement_counts)
    print("\n")

