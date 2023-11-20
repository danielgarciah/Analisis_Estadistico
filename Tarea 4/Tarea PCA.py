# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 18:31:28 2023

@author: danie
"""

import mglearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


#%% importar bases de datos
humidity = pd.read_csv('humidity.csv')
pressure = pd.read_csv('pressure.csv')
temperature = pd.read_csv('temperature.csv')
wind_direction = pd.read_csv('wind_direction.csv')
wind_speed = pd.read_csv('wind_speed.csv')

humidity.name = "humidity"
pressure.name = "pressure"
temperature.name = "temperature"
wind_direction.name = "wind_direction"
wind_speed.name = "wind_speed"


for db in [humidity, pressure, temperature, wind_direction, wind_speed]:
    cols = [i + '_' + db.name for i in db.columns[1:]]
    db.columns = ['datetime'] + cols
    

data = humidity.merge(pressure, on='datetime', how='outer')\
           .merge(temperature, on='datetime', how='outer')\
           .merge(wind_direction, on='datetime', how='outer')\
           .merge(wind_speed, on='datetime', how='outer')

data.set_index('datetime', inplace=True)
data.fillna(0, inplace=True)

cor = data.corr()

#%% Escalando los datos
scaler = StandardScaler()
scaler.fit(data)
scaled_data = scaler.transform(data)

#%% Algoritmo pca
pca = PCA()
pca.fit(scaled_data)

# Ponderación de los componentes principales (vectores propios)
pca_score = pd.DataFrame(data=pca.components_, columns=data.columns,)

pca_data = pca.transform(scaled_data)

#%%Porcentaje de varianza explicada por cada componente principal proporciona
per_var = np.round(pca.explained_variance_ratio_*100, decimals=1)

plt.figure(figsize=(10, 5))
bars=plt.bar(np.arange(len(per_var)), per_var, alpha=0.7)
for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), va='bottom', ha='center', fontsize=10)
plt.xlabel('Componente Principal')
plt.ylabel(' % Varianza Explicada')
plt.xticks(np.arange(len(per_var)))
plt.title('Varianza Explicada por Cada Componente Principal')
plt.show()

#%% Porcentaje de varianza acumulado de los componentes
porcent_acum = np.cumsum(per_var) 

plt.figure()
plt.plot(porcent_acum)
plt.plot([80]*180)
plt.xlabel('Número de componentes')
plt.ylabel('Varianza (%)')  # for each component
plt.title('Porcentaje de varianza acumulada')
plt.show()

#%% Eligiendo número de variables
per_var_df = pd.DataFrame(per_var)
per_var_df['sum'] = per_var.cumsum()

final_data = pd.DataFrame(pca_data).iloc[:, :per_var_df[per_var_df['sum']>=80].index[0]+1]