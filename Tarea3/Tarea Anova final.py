# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 17:19:28 2023

@author: danie
"""

import pandas as pd
import numpy as np
import researchpy as rp
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.stats.multicomp as mc
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
from scipy.stats import kruskal
from scikit_posthocs import posthoc_dunn

#%% Cargar y balancer datos
data = pd.read_csv('HouseListings.csv' , encoding='latin-1')

sns.countplot(data=data, x='Province')
plt.xticks(rotation=90)
plt.show()

print(data.Province.value_counts())

size = data.Province.value_counts().min()
balanced_list = []
for state in data.Province.unique():
    balanced = data[data['Province'] == state].sample(size)
    balanced_list.append(balanced)
    
balanced = pd.concat(balanced_list)

sns.countplot(data=balanced, x='Province')
plt.xticks(rotation=90)
plt.show()

res = rp.summary_cont(balanced['Price'].groupby(balanced['Province']))
balanced.boxplot('Price',by='Province',rot=90)

#%% Cumplimiento de supuestos

# Normalidad
#Normalidad prueba de Shapiro-Wilk
#Ho: Normalidad(p>0.05)
#H1: No normalidad (p<0.05)
#Normalidad en las variables
print(pg.normality(balanced, dv='Price', group='Province'))
          
#Homocedasticidad prueba de Levene (sin normalidad)
#Ho: Homocedasticidad (p>0.05)
#H1: No Homocedasticidad (p<0.05)
print(pg.homoscedasticity(balanced, dv='Price', 
                    group='Province',method='levene'))

#%% KrustalWallis
# Crea listas o arrays con los datos de cada especialidad
for i in data.Province.unique():
    name = i.replace(" ","_")
    exec(f'data_{name} = balanced[balanced["Province"] == i]["Price"]')

# Realiza la prueba de Kruskal-Wallis
stat, p_value = kruskal(data_Ontario, data_British_Columbia, data_Alberta,
                        data_Manitoba, data_Saskatchewan, data_Newfoundland_and_Labrador,
                        data_New_Brunswick, data_Quebec, data_Nova_Scotia)

# Imprime los resultados de Kruskal-Wallis
print(f"Estadístico de prueba (Kruskal-Wallis): {stat}")
print(f"Valor p (Kruskal-Wallis): {p_value}")

# Comprueba si el resultado de Kruskal-Wallis es significativo
alpha = 0.05  # Nivel de significancia
if p_value < alpha:
    print("Hay diferencias significativas entre Province.")
    # Realiza un análisis post hoc de Dunn
    posthoc_result = posthoc_dunn(balanced, val_col='Price', group_col='Province', p_adjust='bonferroni')
else:
    print("No hay diferencias significativas entre Province.")
    
posthoc_result_bool = pd.DataFrame(np.where(posthoc_result<alpha, 'Diferencia', 'No Diferencia'))
posthoc_result_bool.index = posthoc_result.index
posthoc_result_bool.columns = posthoc_result.columns
    
#%% Intentando ANOVA eligiendo sólo los Province y variable que cumplen con Homocedasticidad
data = pd.read_csv('HouseListings.csv' , encoding='latin-1')

data = data[(data['Province']=='Alberta')|(data['Province']=='British Columbia')|\
            (data['Province']=='Quebec')]

sns.countplot(data=data, x='Province')
plt.show()

print(data.Province.value_counts())

size = data.Province.value_counts().min()
balanced_list = []
for state in data.Province.unique():
    balanced = data[data['Province'] == state].sample(size)
    balanced_list.append(balanced)
    
balanced = pd.concat(balanced_list)

sns.countplot(data=balanced, x='Province')
plt.xticks(rotation=90)
plt.show()

res = rp.summary_cont(balanced['Median_Family_Income'].groupby(balanced['Province']))
balanced.boxplot('Median_Family_Income',by='Province',rot=90)

#%% Cumplimiento de supuestos

# Normalidad
#Normalidad prueba de Shapiro-Wilk
#Ho:Normalidad(p>0.05)
#H1: No normalidad (p<0.05)
#Normalidad en las variables
print(pg.normality(balanced, dv='Median_Family_Income', group='Province'))
          
#Homocedasticidad prueba de Levene (sin normalidad)
#Ho:Homocedasticidad (p>0.05)
#H1: No Homocedasticidad (p<0.05)
print(pg.homoscedasticity(balanced, dv='Median_Family_Income', 
                    group='Province',method='levene'))

#%% One way ANOVA
model =ols('Median_Family_Income ~ Province', data=balanced).fit()
anova_table = sm.stats.anova_lm(model,typ=2)
print(anova_table)

#%%Comparación múltiple Prueba de Tukey
comp = mc.MultiComparison(balanced['Median_Family_Income'],balanced['Province'])
post_hoc_res = comp.tukeyhsd()
print(post_hoc_res.summary())

#%% Two ways ANOVA

# Primero hacer cuantitativa la variable "Number_Baths"
valores = [-999, 3, 6, 9, 999]
categorias = ['0-3', '3-6', '6-9', '9+']
balanced['Baths_bracket'] = pd.cut(x=balanced['Number_Baths'], bins=valores, labels=categorias)

#Ho:m1=m2=m3+.... (p>0.05)
#H1: mi dif mj (p<0.05)
model =ols('Median_Family_Income ~ Province + Baths_bracket+Province:Baths_bracket ',
           data=balanced).fit()
anova_table = sm.stats.anova_lm(model,typ=2)
print(anova_table)

#%% Prueba de Tukey (HSD)
interaction_groups = "Province" + balanced.Province.astype(str) + " & " + "Baths_bracket" + balanced.Baths_bracket.astype(str)
comp = mc.MultiComparison(balanced["Median_Family_Income"], interaction_groups)
post_hoc_res = comp.tukeyhsd()
print(post_hoc_res.summary())


