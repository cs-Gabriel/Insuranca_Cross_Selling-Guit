# Bibliotecas
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

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

#%% Vamos tratar os nulls do teste e do treinamento

# Preencher valores ausentes em colunas numéricas com a média
df.fillna(df.select_dtypes(include=['number']).mean(), inplace=True)
test.fillna(test.select_dtypes(include=['number']).mean(), inplace=True)

# Preencher valores ausentes em colunas categóricas com 'Desconhecido'
df.fillna(df.select_dtypes(include=['object']).mode().iloc[0], inplace=True)
test.fillna(test.select_dtypes(include=['object']).mode().iloc[0], inplace=True)

#%% verificar se possui valores duplicados

# Não encontamos valores duplicados
duplicados = df.duplicated()
print(duplicados)

#%% Separando em variaveis e targuet 
X = df.drop(columns=['Price'])
y = df['Price']

#%% Aplicando One-Hot Encoding
# Concatenando treino e teste 
X_total = pd.concat([X, test], axis=0)

# Aplicando One-Hot Encoding no conjunto completo
X_encoded_total = pd.get_dummies(X_total)


# Separando novamente em treino e teste, garantindo as mesmas colunas
X_encoded_test = X_encoded_total.iloc[len(X):, :].reindex(columns=X_encoded_total.iloc[:len(X), :].columns, fill_value=0)
X_encoded = X_encoded_total.iloc[:len(X), :]

#%% Dividindo os dados em Treino e validação

X_train, X_val, y_train, y_val = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

#%% Normalização
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_encoded_test_scaled = scaler.transform(X_encoded_test)

#%% Criando os modelos

modelos = {
    "XGBoost": XGBRegressor( device='cpu', max_depth=5, n_estimators= 1000, learning_rate=0.015, random_state=42),
    "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
    "LightGBM": LGBMRegressor(n_estimators=100, learning_rate=0.015, max_depth=5, random_state=42)           
}

resultados = {}
#%% Treinando os modelos

for nome, modelo in modelos.items():
    print(f"\nTreinando {nome}...")
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    resultados[nome] = rmse
    print(f'{nome} - RMSE: {rmse: .04f}')
    
    
#%%Resultados
melhor_modelo = min(resultados, key=resultados.get)
print(f'\nMelhor modelo: {melhor_modelo} com RMSE de {resultados[melhor_modelo]:.4f}')

#%% Vizualização dos Resultados

plt.figure(figsize=(8, 6))
sns.barplot(x=list(resultados.keys()), y=list(resultados.values()))
plt.title('RMSE dos Modelos')
plt.ylabel('RMSE')
plt.show()

#%% Executando a previsão com os dados de test
""" Agora Sabemos que o modelo XGBoost teve menor taxa de erro, então vamos executalo na base de dados de teste
 fornecida pelo kaggle para contruir o arquivo que sera enviado para competição, para isso temos que fazer o mesmo tratamento de dados
 que foi feito para a base de treino na base de test"""
 
 
#%% Agora vamos prever utilizando o modelo do XGBoost

y_pred_test = modelos[melhor_modelo].predict(X_encoded_test_scaled)

#%%Salvando par subir no kaggle

submission = pd.DataFrame({'id': test['id'], 'Price': y_pred_test})
submission.to_csv('submission.csv', index=False)