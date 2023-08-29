# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 21:47:08 2023

@author: daniel
"""

# Import libraries
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

#%% Import Database
data = pd.read_csv('happyscore_income.csv')

#%% Analyze Data
print(data.info())

# Variables importantes
# avg_satisfaction: Average Satisfaction by Country
# avg_income: Average Income by Country
# income_inequality: Gap between individuals income by Country
# happyScore: Score of Happyness by Country
# GDP: Gross Domestic Product

final = data[['avg_satisfaction', 'avg_income', 'income_inequality', 'happyScore', 'GDP']]

#%% Análisis estadístico unidimensional
Summary = final.describe()

print('HappyScore mean: ', round(Summary.loc['mean', 'happyScore'],2),
      ', HappyScore median: ', round(Summary.loc['50%', 'happyScore'],2),
      ', Standar Deviation: ', round(Summary.loc['std', 'happyScore'],2))

print('GDP mean: ', round(Summary.loc['mean', 'GDP'],2),
      ', GDP median: ', round(Summary.loc['50%', 'GDP'],2),
      ', GDP Deviation: ', round(Summary.loc['std', 'GDP'],2))

#%% Análisis estadístico bidimensional
cov_matrix = final.cov()
corr_matrix = final.corr()

plt.figure()
sns.heatmap(corr_matrix, annot=True, cmap='flare')
plt.title("Correlation Matrix")
plt.show()

# Altamente correlacionados HappyScore con GDP y con Avg_Income con una correlación positiva
# De igual manera Avg_income con avg_satisfaction son áltamente relacionados con un signo positivo.
# Esto nos dice que el nivel de felicidad tiene una relación lineal con el nivel de ingresos de cada país.

# Algo interesante es que la variable income_inequality tiene una relación inversa con todas las variables.
# Sobretodo destacar que tiene una relación inversa de casi 20% con el HappyScore.
# Lo que nos lleva a la conclusión que entre mayor sea el ingreso, y menos inequidad haya en la población
# mayor será la felicidad de dicha población.
