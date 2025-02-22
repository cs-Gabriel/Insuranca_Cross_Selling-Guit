# Bibliotecas

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor

from sklearn.model_selection import RandomizedSearchCV, KFold

from sklearn.metrics import mean_squared_error

#%% Importando Data Frame

#Lendo as os dois DF de traine fornecido pelo kaggle
train0= pd.read_csv('train.csv')
train1= pd.read_csv('training_extra.csv')

# Já vou concatenar os dois de treino
df= pd.concat([train0, train1], axis=0, ignore_index=True)

#Lendo o Test fornecido
test= pd.read_csv('test.csv')

#%% Exploração dos dados

# .info prar observar os types
df.info()

# .describe() para ter dimenção do DF que estamos trabalhando
df.describe()

# .head() para inciar a izualização do dados
df.head()

# Verificando a prenença de null
nulos = df.isnull().sum()

#%% Vizaualidando os Null

nulos.plot(kind='bar')
plt.title('Número de Valores Nulos por Coluna')
plt.xlabel('Colunas')
plt.ylabel('Número de Nulos')
plt.show()

#%% Outro grafico para vizualizar os null

plt.figure(figsize=(20,10))
sns.heatmap(df.isnull())
plt.show()

#%% Vamos tratar os nulls

# Preencher valores ausentes em colunas numéricas com a média
df.fillna(df.select_dtypes(include=['number']).mean(), inplace=True)

# Preencher valores ausentes em colunas categóricas com 'Desconhecido'
df.fillna(df.select_dtypes(include=['object']).mode().iloc[0], inplace=True)

#%% verificar se possui valores duplicados

# Não encontamos valores duplicados
duplicados = df.duplicated()
print(duplicados)

#%% Tranformando as variaveis em numeros
df_n = df

label_encoder = LabelEncoder()


for col in df_n.select_dtypes(include=['object']).columns:
    df_n[col] = label_encoder.fit_transform(df_n[col])

# Exibindo o DataFrame transformado
print(df_n.head())

#%% Vizualização dos dados

#Vamos criar a variavel "Target" que é nossa resposta, para analizar quais variaveis estão mais relacionadas
target = df['Price']

# Calcula a matriz de correlação
corr_matrix = df_n.corr()

# Plota uma matriz com as correlações
corr_matrix = df_n.corr()
plt.figure(figsize=(15, 10))
sns.heatmap(corr_matrix, 
            annot=True, 
            linewidths=0.5, 
            fmt= ".2f", 
            cmap="YlGnBu");
# A matrix não apresnta forte correlações

#%%
fig,ax=plt.subplots(4,2,figsize=(20,20))
ax=ax.flatten()
i=0
for col in df.columns[df.dtypes=='object']:
    if col !='Brand':
        sns.countplot(data=df,x=col,ax=ax[i],hue='Brand')
        i+=1
sns.countplot(data=df,x='Compartments',ax=ax[i],hue='Brand')
ax[7].axis('off')
plt.tight_layout()
plt.show()