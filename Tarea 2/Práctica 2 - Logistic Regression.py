# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 14:31:52 2023

@author: danie
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.stats.api as sms
from scipy import stats
from statsmodels.compat import lzip
from sklearn.model_selection import train_test_split

data = pd.read_csv("song_data.csv")

correlation = data.corr()
plt.figure(figsize=(12,12))
sns.heatmap(correlation, annot=True, cmap='flare')
plt.title("Correlation Matrix")
plt.show()

X = data[['danceability']]
Y = data['song_popularity'] 

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

X_train = sm.add_constant(X_train)
modelo = sm.OLS(Y_train, X_train).fit()
print(modelo.summary())

"""
R-squared: El 1.1% de la popularidad de una canción se explica con la danceabilidad.
Prob(F-statistics): se rechaza hipótesis nula, por lo tanto, nuestra variable tienen
significancia en el modelo y explica mas que el azar.
No se rechaza ni B0 ni B1.
Por la prueba de Omnibus y Jarque-Bera podemos ver que los errores no siguen una distribución normal.
La campana está sezgada a la izquierda y tienen una curtosis de 2.935
Al ver la prueba de Durbin Watson podemos ver que los datos tienen casi una homocedastidad perfecta,
al estar bastante cerca de 2.
"""

X_test = sm.add_constant(X_test)
Y_pred = modelo.predict(X_test)

error = Y_test - Y_pred
plt.scatter(X_test.iloc[:,1], Y_test, label='Valores reales')
plt.plot(X_test.iloc[:,1], Y_pred, color='r', label='Predicciones', linewidth=3)
plt.title('Regresión Lineal')
plt.xlabel('Song Popularity')
plt.ylabel('Danceability')
plt.legend()
plt.show()

# Homocedasticidad
plt.scatter(Y_pred, error, marker='*', alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Valores Ajustados')
plt.ylabel('Residuos')
plt.title('Homocedasticidad')
plt.show()

#Breusch-Pagan
#H0: Homocedasticidad (p>0.05)
#H1: No homocedasticidad (p<0.05)
names=['Lagrange multiplier statistic', 'p-value',
        'f-value', 'f p-value']
test = sms.het_breuschpagan(modelo.resid, X_train)
print(lzip(names, test))
"""
Con esta prueba vemos que no hay homocedasticidad.
"""

# Normalidad
plt.figure()
plt.hist(modelo.resid)
plt.show()

# QQ Plot
qq_plot = sm.qqplot(modelo.resid, line='45', fit=True)
plt.title("QQ Plot de los Residuos")
plt.xlabel("Cuantiles Teóricos")
plt.ylabel("Cuantiles de los Residuos")
plt.show()

"""
No es normal
"""

# Shapiro-Wilk)
#Ho: Normalidad (p>0.05)
#H1: No normalidad (p<0.05)
names=[' Statistic', 'p-value']
test=stats.shapiro(modelo.resid)
print(lzip(names,test))
"""
No es normal
"""

