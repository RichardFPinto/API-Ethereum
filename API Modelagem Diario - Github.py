#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install Historic-Crypto
#!pip install pandas_datareader
#!pip install yfinance


# In[43]:


from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager
import pandas as pd
import time
from datetime import datetime
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
import math
from winsound import Beep
import yfinance as yf
from tensorflow.keras.models import load_model


# # Dados
# ## Conseguindo os dados
# ### Método 1

# In[12]:


# Pegando os dados do Yahoo Finance
hoje = datetime.today().strftime('%Y-%m-%d')
dados = yf.download('ETH-USD','2022-01-01', hoje)
dados


# ### Método 2

# In[21]:


# Caso por algum motivo o Yahoo tiver fora do ar, dados conseguido de outro método
#dados = pd.read_csv('C:/Users/rfp20/Desktop/Tr/dados historicos.csv',sep = ',', decimal='.')


# ## Organizando e preparando os dados para a modelagem

# In[23]:


dados = dados.dropna()
base = dados
n = base.shape[0] - 22
base, base_test= np.split(base, [n])
base_open = base.iloc[:, 1:2].values
base_high = base.iloc[:, 2:3].values
base_low = base.iloc[:, 3:4].values
base_close = base.iloc[:, 4:5].values


base_test_open = base_test.iloc[:, 1:2].values
base_test_high = base_test.iloc[:, 2:3].values
base_test_low = base_test.iloc[:, 3:4].values
base_test_close = base_test.iloc[:, 4:5].values


# In[26]:


normalizador = MinMaxScaler(feature_range=(0,1))

base_open_normalizada = normalizador.fit_transform(base_open)
base_high_normalizada = normalizador.fit_transform(base_high)
base_low_normalizada = normalizador.fit_transform(base_low)
base_close_normalizada = normalizador.fit_transform(base_close)

normalizador_previsao_open = MinMaxScaler(feature_range=(0,1))
normalizador_previsao_open.fit_transform(base_open[:,0:1])

normalizador_previsao_high = MinMaxScaler(feature_range=(0,1))
normalizador_previsao_high.fit_transform(base_high[:,0:1])

normalizador_previsao_low = MinMaxScaler(feature_range=(0,1))
normalizador_previsao_low.fit_transform(base_low[:,0:1])

normalizador_previsao_close = MinMaxScaler(feature_range=(0,1))
normalizador_previsao_close.fit_transform(base_close[:,0:1])

previsores_open = []
preco_real_open = []

for i in range(90, base_open.shape[0]):
    previsores_open.append(base_open_normalizada[i-90:i, 0])
    preco_real_open.append(base_open_normalizada[i, 0])

previsores_high = []
preco_real_high = []

for i in range(90, base_high.shape[0]):
    previsores_high.append(base_high_normalizada[i-90:i, 0])
    preco_real_high.append(base_high_normalizada[i, 0])
    
previsores_low = []
preco_real_low = []

for i in range(90, base_low.shape[0]):
    previsores_low.append(base_low_normalizada[i-90:i, 0])
    preco_real_low.append(base_low_normalizada[i, 0])

previsores_close = []
preco_real_close = []

for i in range(90, base_close.shape[0]):
    previsores_close.append(base_close_normalizada[i-90:i, 0])
    preco_real_close.append(base_close_normalizada[i, 0])
    
previsores_open, preco_real_open = np.array(previsores_open), np.array(preco_real_open)
previsores_high, preco_real_high = np.array(previsores_high), np.array(preco_real_high)
previsores_low, preco_real_low = np.array(previsores_low), np.array(preco_real_low)
previsores_close, preco_real_close = np.array(previsores_close), np.array(preco_real_close)

previsores_open = np.reshape(previsores_open, (previsores_open.shape[0], previsores_open.shape[1],1))
previsores_high = np.reshape(previsores_high, (previsores_high.shape[0], previsores_high.shape[1],1))
previsores_low = np.reshape(previsores_low, (previsores_low.shape[0], previsores_low.shape[1],1))
previsores_close = np.reshape(previsores_close, (previsores_close.shape[0], previsores_close.shape[1],1))


# In[27]:


base_test_open = pd.DataFrame(base_test_open)
base_test_high = pd.DataFrame(base_test_high)
base_test_low = pd.DataFrame(base_test_low)
base_test_close = pd.DataFrame(base_test_close)

base_open = pd.DataFrame(base_open)
base_high = pd.DataFrame(base_high)
base_low = pd.DataFrame(base_low)
base_close = pd.DataFrame(base_close)


# In[28]:


base_completa_open = pd.concat((base_open,base_test_open), axis = 0)
base_completa_high = pd.concat((base_high,base_test_high), axis = 0)
base_completa_low = pd.concat((base_low,base_test_low), axis = 0)
base_completa_close = pd.concat((base_close,base_test_close), axis = 0)


# In[29]:


entradas_open = base_completa_open[len(base_completa_open) - len(base_test_open) - 90:].values
entradas_open = entradas_open.reshape(-1,1)
entradas_open = normalizador.transform(entradas_open)

entradas_high = base_completa_high[len(base_completa_high) - len(base_test_high) - 90:].values
entradas_high = entradas_high.reshape(-1,1)
entradas_high = normalizador.transform(entradas_high)

entradas_low = base_completa_low[len(base_completa_low) - len(base_test_low) - 90:].values
entradas_low = entradas_low.reshape(-1,1)
entradas_low = normalizador.transform(entradas_low)

entradas_close = base_completa_close[len(base_completa_close) - len(base_test_close) - 90:].values
entradas_close = entradas_close.reshape(-1,1)
entradas_close = normalizador.transform(entradas_close)

X_teste_open = []
for i in range(90, 112):
    X_teste_open.append(entradas_open[i-90:i, 0])
X_teste_open = np.array(X_teste_open)
X_teste_open = np.reshape(X_teste_open, (X_teste_open.shape[0],X_teste_open.shape[1],1))

X_teste_high = []
for i in range(90, 112):
    X_teste_high.append(entradas_high[i-90:i, 0])
X_teste_high = np.array(X_teste_high)
X_teste_high = np.reshape(X_teste_high, (X_teste_high.shape[0],X_teste_high.shape[1],1))

X_teste_low = []
for i in range(90, 112):
    X_teste_low.append(entradas_low[i-90:i, 0])
X_teste_low = np.array(X_teste_low)
X_teste_low = np.reshape(X_teste_low, (X_teste_low.shape[0],X_teste_low.shape[1],1))

X_teste_close = []
for i in range(90, 112):
    X_teste_close.append(entradas_close[i-90:i, 0])
X_teste_close = np.array(X_teste_close)
X_teste_close = np.reshape(X_teste_close, (X_teste_close.shape[0],X_teste_close.shape[1],1))


# # Modelagem
# Foram utilizados 4 modelos para de LSTM com pequenas mudanças,cada um foi utilizado na abertura, valor máximo, valor mínimo e fechamento. Foi avaliada a precisão utilizando o RMSE e o MAE, com mais enfase nos valores do RMSE, pois muitos autores indicam esse o avaliador de precisão, entre esses autores Morettin.
# 
# Foi utilizado abertura, valor máximo, valor mínimo e fechamento, para caso se necessário a utilização de métodos como Garman Klass para a volatilidade realizada do valor previsto.

# In[81]:


regressor1_open = Sequential()
regressor1_open.add(LSTM(units = 100, return_sequences = True, input_shape = (previsores_open.shape[1], 1)))
regressor1_open.add(Dropout(0.3))

regressor1_open.add(LSTM(units = 50, return_sequences = True))
regressor1_open.add(Dropout(0.3))

regressor1_open.add(LSTM(units = 50, return_sequences = True))
regressor1_open.add(Dropout(0.3))

regressor1_open.add(LSTM(units = 50))
regressor1_open.add(Dropout(0.3))
regressor1_open.add(Dense(units = 1, activation = 'sigmoid'))

regressor1_open.compile(optimizer = 'adam', loss = 'mean_squared_error',metrics = ['mean_absolute_error'])

regressor1_open.fit(previsores_open, preco_real_open, epochs = 200, batch_size = 32)


# In[83]:


train_score = regressor1_open.evaluate(previsores_open, preco_real_open, verbose=0)
rmse = (train_score[1] ** 1/2)
print('Pontuação de Teste:', train_score[0], 'MSE', rmse,'RMSE')


# In[84]:


regressor1_high = Sequential()
regressor1_high.add(LSTM(units = 100, return_sequences = True, input_shape = (previsores_high.shape[1], 1)))
regressor1_high.add(Dropout(0.3))

regressor1_high.add(LSTM(units = 50, return_sequences = True))
regressor1_high.add(Dropout(0.3))

regressor1_high.add(LSTM(units = 50, return_sequences = True))
regressor1_high.add(Dropout(0.3))

regressor1_high.add(LSTM(units = 50))
regressor1_high.add(Dropout(0.3))
regressor1_high.add(Dense(units = 1, activation = 'sigmoid'))

regressor1_high.compile(optimizer = 'adam', loss = 'mean_squared_error',metrics = ['mean_absolute_error'])

regressor1_high.fit(previsores_high, preco_real_high, epochs = 200, batch_size = 32)


# In[85]:


train_score = regressor1_high.evaluate(previsores_high, preco_real_high, verbose=0)
rmse = (train_score[1] ** 1/2)
print('Pontuação de Teste:', train_score[0], 'MSE', rmse,'RMSE')


# In[86]:


regressor1_low = Sequential()
regressor1_low.add(LSTM(units = 100, return_sequences = True, input_shape = (previsores_low.shape[1], 1)))
regressor1_low.add(Dropout(0.3))

regressor1_low.add(LSTM(units = 50, return_sequences = True))
regressor1_low.add(Dropout(0.3))

regressor1_low.add(LSTM(units = 50, return_sequences = True))
regressor1_low.add(Dropout(0.3))

regressor1_low.add(LSTM(units = 50))
regressor1_low.add(Dropout(0.3))
regressor1_low.add(Dense(units = 1, activation = 'sigmoid'))

regressor1_low.compile(optimizer = 'adam', loss = 'mean_squared_error',metrics = ['mean_absolute_error'])

regressor1_low.fit(previsores_low, preco_real_low, epochs = 200, batch_size = 32)


# In[88]:


train_score = regressor1_low.evaluate(previsores_low, preco_real_low, verbose=0)
rmse = (train_score[1] ** 1/2)
print('Pontuação de Teste:', train_score[0], 'MSE', rmse,'RMSE')


# In[89]:


regressor1_close = Sequential()
regressor1_close.add(LSTM(units = 100, return_sequences = True, input_shape = (previsores_close.shape[1], 1)))
regressor1_close.add(Dropout(0.3))

regressor1_close.add(LSTM(units = 50, return_sequences = True))
regressor1_close.add(Dropout(0.3))

regressor1_close.add(LSTM(units = 50, return_sequences = True))
regressor1_close.add(Dropout(0.3))

regressor1_close.add(LSTM(units = 50))
regressor1_close.add(Dropout(0.3))
regressor1_close.add(Dense(units = 1, activation = 'sigmoid'))

regressor1_close.compile(optimizer = 'adam', loss = 'mean_squared_error',metrics = ['mean_absolute_error'])

regressor1_close.fit(previsores_close, preco_real_close, epochs = 200, batch_size = 32)


# In[90]:


train_score = regressor1_close.evaluate(previsores_close, preco_real_close, verbose=0)
rmse = (train_score[1] ** 1/2)
print('Pontuação de Teste:', train_score[0], 'MSE', rmse,'RMSE')


# In[91]:


regressor2_open = Sequential()
regressor2_open.add(LSTM(units = 100, return_sequences = True, input_shape = (previsores_open.shape[1], 1)))
regressor2_open.add(Dropout(0.3))

regressor2_open.add(LSTM(units = 50, return_sequences = True))
regressor2_open.add(Dropout(0.3))

regressor2_open.add(LSTM(units = 50, return_sequences = True))
regressor2_open.add(Dropout(0.3))

regressor2_open.add(LSTM(units = 50))
regressor2_open.add(Dropout(0.3))

regressor2_open.add(Dense(units = 1, activation = 'sigmoid'))

regressor2_open.compile(optimizer = 'rmsprop', loss = 'mean_squared_error',metrics = ['mean_absolute_error'])
regressor2_open.fit(previsores_open, preco_real_open, epochs = 200, batch_size = 32)


# In[92]:


train_score = regressor2_open.evaluate(previsores_open, preco_real_open, verbose=0)
rmse = (train_score[1] ** 1/2)
print('Pontuação de Teste:', train_score[0], 'MSE', rmse,'RMSE')


# In[93]:


regressor2_high = Sequential()
regressor2_high.add(LSTM(units = 100, return_sequences = True, input_shape = (previsores_high.shape[1], 1)))
regressor2_high.add(Dropout(0.3))

regressor2_high.add(LSTM(units = 50, return_sequences = True))
regressor2_high.add(Dropout(0.3))

regressor2_high.add(LSTM(units = 50, return_sequences = True))
regressor2_high.add(Dropout(0.3))

regressor2_high.add(LSTM(units = 50))
regressor2_high.add(Dropout(0.3))

regressor2_high.add(Dense(units = 1, activation = 'sigmoid'))

regressor2_high.compile(optimizer = 'rmsprop', loss = 'mean_squared_error',metrics = ['mean_absolute_error'])
regressor2_high.fit(previsores_high, preco_real_high, epochs = 200, batch_size = 32)


# In[94]:


train_score = regressor2_high.evaluate(previsores_high, preco_real_high, verbose=0)
rmse = (train_score[1] ** 1/2)
print('Pontuação de Teste:', train_score[0], 'MSE', rmse,'RMSE')


# In[95]:


regressor2_low = Sequential()
regressor2_low.add(LSTM(units = 100, return_sequences = True, input_shape = (previsores_low.shape[1], 1)))
regressor2_low.add(Dropout(0.3))

regressor2_low.add(LSTM(units = 50, return_sequences = True))
regressor2_low.add(Dropout(0.3))

regressor2_low.add(LSTM(units = 50, return_sequences = True))
regressor2_low.add(Dropout(0.3))

regressor2_low.add(LSTM(units = 50))
regressor2_low.add(Dropout(0.3))

regressor2_low.add(Dense(units = 1, activation = 'sigmoid'))

regressor2_low.compile(optimizer = 'rmsprop', loss = 'mean_squared_error',metrics = ['mean_absolute_error'])
regressor2_low.fit(previsores_low, preco_real_low, epochs = 200, batch_size = 32)


# In[96]:


train_score = regressor2_low.evaluate(previsores_low, preco_real_low, verbose=0)
rmse = (train_score[1] ** 1/2)
print('Pontuação de Teste:', train_score[0], 'MSE', rmse,'RMSE')


# In[97]:


regressor2_close = Sequential()
regressor2_close.add(LSTM(units = 100, return_sequences = True, input_shape = (previsores_close.shape[1], 1)))
regressor2_close.add(Dropout(0.3))

regressor2_close.add(LSTM(units = 50, return_sequences = True))
regressor2_close.add(Dropout(0.3))

regressor2_close.add(LSTM(units = 50, return_sequences = True))
regressor2_close.add(Dropout(0.3))

regressor2_close.add(LSTM(units = 50))
regressor2_close.add(Dropout(0.3))

regressor2_close.add(Dense(units = 1, activation = 'sigmoid'))

regressor2_close.compile(optimizer = 'rmsprop', loss = 'mean_squared_error',metrics = ['mean_absolute_error'])
regressor2_close.fit(previsores_close, preco_real_close, epochs = 200, batch_size = 32)


# In[98]:


train_score = regressor2_close.evaluate(previsores_close, preco_real_close, verbose=0)
rmse = (train_score[1] ** 1/2)
print('Pontuação de Teste:', train_score[0], 'MSE', rmse,'RMSE')


# In[99]:


regressor3_open = Sequential()
regressor3_open.add(LSTM(units = 100, return_sequences = True, input_shape = (previsores_open.shape[1], 1)))
regressor3_open.add(Dropout(0.3))

regressor3_open.add(LSTM(units = 50, return_sequences = True))
regressor3_open.add(Dropout(0.3))

regressor3_open.add(LSTM(units = 50, return_sequences = True))
regressor3_open.add(Dropout(0.3))

regressor3_open.add(LSTM(units = 50))
regressor3_open.add(Dropout(0.3))

regressor3_open.add(Dense(units = 1, activation = 'linear'))

regressor3_open.compile(optimizer = 'adam', loss = 'mean_squared_error',metrics = ['mean_absolute_error'])
regressor3_open.fit(previsores_open, preco_real_open, epochs = 200, batch_size = 32)


# In[100]:


train_score = regressor3_open.evaluate(previsores_open, preco_real_open, verbose=0)
rmse = (train_score[1] ** 1/2)
print('Pontuação de Teste:', train_score[0], 'MSE', rmse,'RMSE')


# In[101]:


regressor3_high = Sequential()
regressor3_high.add(LSTM(units = 100, return_sequences = True, input_shape = (previsores_high.shape[1], 1)))
regressor3_high.add(Dropout(0.3))

regressor3_high.add(LSTM(units = 50, return_sequences = True))
regressor3_high.add(Dropout(0.3))

regressor3_high.add(LSTM(units = 50, return_sequences = True))
regressor3_high.add(Dropout(0.3))

regressor3_high.add(LSTM(units = 50))
regressor3_high.add(Dropout(0.3))

regressor3_high.add(Dense(units = 1, activation = 'linear'))

regressor3_high.compile(optimizer = 'adam', loss = 'mean_squared_error',metrics = ['mean_absolute_error'])
regressor3_high.fit(previsores_high, preco_real_high, epochs = 200, batch_size = 32)


# In[102]:


train_score = regressor3_high.evaluate(previsores_high, preco_real_high, verbose=0)
rmse = (train_score[1] ** 1/2)
print('Pontuação de Teste:', train_score[0], 'MSE', rmse,'RMSE')


# In[103]:


regressor3_low = Sequential()
regressor3_low.add(LSTM(units = 100, return_sequences = True, input_shape = (previsores_low.shape[1], 1)))
regressor3_low.add(Dropout(0.3))

regressor3_low.add(LSTM(units = 50, return_sequences = True))
regressor3_low.add(Dropout(0.3))

regressor3_low.add(LSTM(units = 50, return_sequences = True))
regressor3_low.add(Dropout(0.3))

regressor3_low.add(LSTM(units = 50))
regressor3_low.add(Dropout(0.3))

regressor3_low.add(Dense(units = 1, activation = 'linear'))

regressor3_low.compile(optimizer = 'adam', loss = 'mean_squared_error',metrics = ['mean_absolute_error'])
regressor3_low.fit(previsores_low, preco_real_low, epochs = 200, batch_size = 32)


# In[104]:


train_score = regressor3_low.evaluate(previsores_low, preco_real_low, verbose=0)
rmse = (train_score[1] ** 1/2)
print('Pontuação de Teste:', train_score[0], 'MSE', rmse,'RMSE')


# In[105]:


regressor3_close = Sequential()
regressor3_close.add(LSTM(units = 100, return_sequences = True, input_shape = (previsores_close.shape[1], 1)))
regressor3_close.add(Dropout(0.3))

regressor3_close.add(LSTM(units = 50, return_sequences = True))
regressor3_close.add(Dropout(0.3))

regressor3_close.add(LSTM(units = 50, return_sequences = True))
regressor3_close.add(Dropout(0.3))

regressor3_close.add(LSTM(units = 50))
regressor3_close.add(Dropout(0.3))

regressor3_close.add(Dense(units = 1, activation = 'linear'))

regressor3_close.compile(optimizer = 'adam', loss = 'mean_squared_error',metrics = ['mean_absolute_error'])
regressor3_close.fit(previsores_close, preco_real_close, epochs = 200, batch_size = 32)


# In[106]:


train_score = regressor3_close.evaluate(previsores_close, preco_real_close, verbose=0)
rmse = (train_score[1] ** 1/2)
print('Pontuação de Teste:', train_score[0], 'MSE', rmse,'RMSE')


# In[107]:


regressor4_open = Sequential()
regressor4_open.add(LSTM(units = 100, return_sequences = True, input_shape = (previsores_open.shape[1], 1)))
regressor4_open.add(Dropout(0.3))

regressor4_open.add(LSTM(units = 50, return_sequences = True))
regressor4_open.add(Dropout(0.3))

regressor4_open.add(LSTM(units = 50, return_sequences = True))
regressor4_open.add(Dropout(0.3))

regressor4_open.add(LSTM(units = 50))
regressor4_open.add(Dropout(0.3))

regressor4_open.add(Dense(units = 1, activation = 'linear'))

regressor4_open.compile(optimizer = 'rmsprop', loss = 'mean_squared_error',metrics = ['mean_absolute_error'])
regressor4_open.fit(previsores_open, preco_real_open, epochs = 200, batch_size = 32)


# In[108]:


train_score = regressor4_open.evaluate(previsores_open, preco_real_open, verbose=0)
rmse = (train_score[1] ** 1/2)
print('Pontuação de Teste:', train_score[0], 'MSE', rmse,'RMSE')


# In[109]:


regressor4_high = Sequential()
regressor4_high.add(LSTM(units = 100, return_sequences = True, input_shape = (previsores_high.shape[1], 1)))
regressor4_high.add(Dropout(0.3))

regressor4_high.add(LSTM(units = 50, return_sequences = True))
regressor4_high.add(Dropout(0.3))

regressor4_high.add(LSTM(units = 50, return_sequences = True))
regressor4_high.add(Dropout(0.3))

regressor4_high.add(LSTM(units = 50))
regressor4_high.add(Dropout(0.3))

regressor4_high.add(Dense(units = 1, activation = 'linear'))

regressor4_high.compile(optimizer = 'rmsprop', loss = 'mean_squared_error',metrics = ['mean_absolute_error'])
regressor4_high.fit(previsores_high, preco_real_high, epochs = 200, batch_size = 32)


# In[110]:


train_score = regressor4_high.evaluate(previsores_high, preco_real_high, verbose=0)
rmse = (train_score[1] ** 1/2)
print('Pontuação de Teste:', train_score[0], 'MSE', rmse,'RMSE')


# In[111]:


regressor4_low = Sequential()
regressor4_low.add(LSTM(units = 100, return_sequences = True, input_shape = (previsores_low.shape[1], 1)))
regressor4_low.add(Dropout(0.3))

regressor4_low.add(LSTM(units = 50, return_sequences = True))
regressor4_low.add(Dropout(0.3))

regressor4_low.add(LSTM(units = 50, return_sequences = True))
regressor4_low.add(Dropout(0.3))

regressor4_low.add(LSTM(units = 50))
regressor4_low.add(Dropout(0.3))

regressor4_low.add(Dense(units = 1, activation = 'linear'))

regressor4_low.compile(optimizer = 'rmsprop', loss = 'mean_squared_error',metrics = ['mean_absolute_error'])
regressor4_low.fit(previsores_low, preco_real_low, epochs = 200, batch_size = 32)


# In[112]:


train_score = regressor4_low.evaluate(previsores_low, preco_real_low, verbose=0)
rmse = (train_score[1] ** 1/2)
print('Pontuação de Teste:', train_score[0], 'MSE', rmse,'RMSE')


# In[113]:


regressor4_close = Sequential()
regressor4_close.add(LSTM(units = 100, return_sequences = True, input_shape = (previsores_close.shape[1], 1)))
regressor4_close.add(Dropout(0.3))

regressor4_close.add(LSTM(units = 50, return_sequences = True))
regressor4_close.add(Dropout(0.3))

regressor4_close.add(LSTM(units = 50, return_sequences = True))
regressor4_close.add(Dropout(0.3))

regressor4_close.add(LSTM(units = 50))
regressor4_close.add(Dropout(0.3))

regressor4_close.add(Dense(units = 1, activation = 'linear'))

regressor4_close.compile(optimizer = 'rmsprop', loss = 'mean_squared_error',metrics = ['mean_absolute_error'])
regressor4_close.fit(previsores_close, preco_real_close, epochs = 200, batch_size = 32)


# In[114]:


train_score = regressor4_close.evaluate(previsores_close, preco_real_close, verbose=0)
rmse = (train_score[1] ** 1/2)
print('Pontuação de Teste:', train_score[0], 'MSE', rmse,'RMSE')


# ## Salvando os modelos

# In[115]:


regressor1_open.save('C:/Users/rfp20/Desktop/Tr/regressor1_open.h5')
regressor2_open.save('C:/Users/rfp20/Desktop/Tr/regressor2_open.h5')
regressor3_open.save('C:/Users/rfp20/Desktop/Tr/regressor3_open.h5')
regressor4_open.save('C:/Users/rfp20/Desktop/Tr/regressor4_open.h5')

regressor1_high.save('C:/Users/rfp20/Desktop/Tr/regressor1_high.h5')
regressor2_high.save('C:/Users/rfp20/Desktop/Tr/regressor2_high.h5')
regressor3_high.save('C:/Users/rfp20/Desktop/Tr/regressor3_high.h5')
regressor4_high.save('C:/Users/rfp20/Desktop/Tr/regressor4_high.h5')

regressor1_low.save('C:/Users/rfp20/Desktop/Tr/regressor1_low.h5')
regressor2_low.save('C:/Users/rfp20/Desktop/Tr/regressor2_low.h5')
regressor3_low.save('C:/Users/rfp20/Desktop/Tr/regressor3_low.h5')
regressor4_low.save('C:/Users/rfp20/Desktop/Tr/regressor4_low.h5')

regressor1_close.save('C:/Users/rfp20/Desktop/Tr/regressor1_close.h5')
regressor2_close.save('C:/Users/rfp20/Desktop/Tr/regressor2_close.h5')
regressor3_close.save('C:/Users/rfp20/Desktop/Tr/regressor3_close.h5')
regressor4_close.save('C:/Users/rfp20/Desktop/Tr/regressor4_close.h5')


# ## Carregando os modelos

# In[33]:


regressor1_open = load_model('C:/Users/rfp20/Desktop/Tr/regressor1_open.h5')
regressor2_open = load_model('C:/Users/rfp20/Desktop/Tr/regressor2_open.h5')
regressor3_open = load_model('C:/Users/rfp20/Desktop/Tr/regressor3_open.h5')
regressor4_open = load_model('C:/Users/rfp20/Desktop/Tr/regressor4_open.h5')

regressor1_high = load_model('C:/Users/rfp20/Desktop/Tr/regressor1_high.h5')
regressor2_high = load_model('C:/Users/rfp20/Desktop/Tr/regressor2_high.h5')
regressor3_high = load_model('C:/Users/rfp20/Desktop/Tr/regressor3_high.h5')
regressor4_high = load_model('C:/Users/rfp20/Desktop/Tr/regressor4_high.h5')

regressor1_low = load_model('C:/Users/rfp20/Desktop/Tr/regressor1_low.h5')
regressor2_low = load_model('C:/Users/rfp20/Desktop/Tr/regressor2_low.h5')
regressor3_low = load_model('C:/Users/rfp20/Desktop/Tr/regressor3_low.h5')
regressor4_low = load_model('C:/Users/rfp20/Desktop/Tr/regressor4_low.h5')

regressor1_close = load_model('C:/Users/rfp20/Desktop/Tr/regressor1_close.h5')
regressor2_close = load_model('C:/Users/rfp20/Desktop/Tr/regressor2_close.h5')
regressor3_close = load_model('C:/Users/rfp20/Desktop/Tr/regressor3_close.h5')
regressor4_close = load_model('C:/Users/rfp20/Desktop/Tr/regressor4_close.h5')


# ### Inicializando algumas variáveis para a próxima parte, criação das notas que serão usadas para aviso de compra e vende posteriormente

# In[116]:


data = []
preco = []

previsao1_open = []
previsao2_open = []
previsao3_open = []
previsao4_open = []
previsao_m_open = []

previsao1_high = []
previsao2_high = []
previsao3_high = []
previsao4_high = []
previsao_m_high = []

previsao1_low = []
previsao2_low = []
previsao3_low = []
previsao4_low = []
previsao_m_low = []

previsao1_close = []
previsao2_close = []
previsao3_close = []
previsao4_close = []
previsao_m_close = []

notes = {'C': 1635,
         'D': 1835,
         'E': 2060,
         'S': 1945,
         'F': 2183,
         'G': 2450,
         'A': 2750,
         'B': 3087,
         ' ': 37}


# ## Criando as previsões de todos os 16 modelos

# In[117]:


# Tratamento para pegar o numero de linhas necessarias
entradas_open = base_completa_open.values
entradas_high = base_completa_high.values
entradas_low = base_completa_low.values
entradas_close = base_completa_close.values
    
# Formato do numpy
entradas_open = entradas_open.reshape(-1, 1)
entradas_high = entradas_high.reshape(-1, 1)
entradas_low = entradas_low.reshape(-1, 1)
entradas_close = entradas_close.reshape(-1, 1)
# Normalizando
entradas_open = normalizador.transform(entradas_open)
        entradas_high = normalizador.transform(entradas_high)
        entradas_low = normalizador.transform(entradas_low)
        entradas_close = normalizador.transform(entradas_close)
        # Criando o loop for
    X_teste_open = []
    X_teste_high = []
    X_teste_low = []
    X_teste_close = []
    # Função para criar as variaveis
    for i in range(90, entradas_open.shape[0]):
        X_teste_open.append(entradas_open[i-90:i, 0])
    # Transformando os dados   
    X_teste_open = np.array(X_teste_open)
    X_teste_open = np.reshape(X_teste_open, (X_teste_open.shape[0], X_teste_open.shape[1], 1))
    
     # Função para criar as variaveis
    for i in range(90, entradas_high.shape[0]):
        X_teste_high.append(entradas_high[i-90:i, 0])
    # Transformando os dados   
    X_teste_high = np.array(X_teste_high)
    X_teste_high = np.reshape(X_teste_high, (X_teste_high.shape[0], X_teste_high.shape[1], 1))
    
     # Função para criar as variaveis
    for i in range(90, entradas_low.shape[0]):
        X_teste_low.append(entradas_low[i-90:i, 0])
    # Transformando os dados   
    X_teste_low = np.array(X_teste_low)
    X_teste_low = np.reshape(X_teste_low, (X_teste_low.shape[0], X_teste_low.shape[1], 1))
    
     # Função para criar as variaveis
    for i in range(90, entradas_close.shape[0]):
        X_teste_close.append(entradas_close[i-90:i, 0])
    # Transformando os dados   
    X_teste_close = np.array(X_teste_close)
    X_teste_close = np.reshape(X_teste_close, (X_teste_close.shape[0], X_teste_close.shape[1], 1))
    
    
    # Realizando a previsão
    
    previsao1_open = regressor1_open.predict(X_teste_open)
    previsao2_open = regressor2_open.predict(X_teste_open)
    previsao3_open = regressor3_open.predict(X_teste_open)
    previsao4_open = regressor4_open.predict(X_teste_open)
    
    previsao1_high = regressor1_high.predict(X_teste_high)
    previsao2_high = regressor2_high.predict(X_teste_high)
    previsao3_high = regressor3_high.predict(X_teste_high)
    previsao4_high = regressor4_high.predict(X_teste_high)
    
    previsao1_low = regressor1_low.predict(X_teste_low)
    previsao2_low = regressor2_low.predict(X_teste_low)
    previsao3_low = regressor3_low.predict(X_teste_low)
    previsao4_low = regressor4_low.predict(X_teste_low)
    
    previsao1_close = regressor1_close.predict(X_teste_close)
    previsao2_close = regressor2_close.predict(X_teste_close)
    previsao3_close = regressor3_close.predict(X_teste_close)
    previsao4_close = regressor4_close.predict(X_teste_close)
    
    # Desnormalizando os dados
    previsao1_open = normalizador.inverse_transform(previsao1_open)
    previsao2_open = normalizador.inverse_transform(previsao2_open)
    previsao3_open = normalizador.inverse_transform(previsao3_open)
    previsao4_open = normalizador.inverse_transform(previsao4_open)
       
    previsao1_high = normalizador.inverse_transform(previsao1_high)
    previsao2_high = normalizador.inverse_transform(previsao2_high)
    previsao3_high = normalizador.inverse_transform(previsao3_high)
    previsao4_high = normalizador.inverse_transform(previsao4_high)
    
    previsao1_low = normalizador.inverse_transform(previsao1_low)
    previsao2_low = normalizador.inverse_transform(previsao2_low)
    previsao3_low = normalizador.inverse_transform(previsao3_low)
    previsao4_low = normalizador.inverse_transform(previsao4_low)
    
    previsao1_close = normalizador.inverse_transform(previsao1_close)
    previsao2_close = normalizador.inverse_transform(previsao2_close)
    previsao3_close = normalizador.inverse_transform(previsao3_close)
    previsao4_close = normalizador.inverse_transform(previsao4_close)
    
    # Ultimo Registro
    
    previsao1_open = float(previsao1_open[len(previsao1_open)-1])
    previsao2_open = float(previsao2_open[len(previsao2_open)-1])
    previsao3_open = float(previsao3_open[len(previsao3_open)-1])
    previsao4_open = float(previsao4_open[len(previsao4_open)-1])
    
    previsao1_high = float(previsao1_high[len(previsao1_high)-1])
    previsao2_high = float(previsao2_high[len(previsao2_high)-1])
    previsao3_high = float(previsao3_high[len(previsao3_high)-1])
    previsao4_high = float(previsao4_high[len(previsao4_high)-1])
    
    previsao1_low = float(previsao1_low[len(previsao1_low)-1])
    previsao2_low = float(previsao2_low[len(previsao2_low)-1])
    previsao3_low = float(previsao3_low[len(previsao3_low)-1])
    previsao4_low = float(previsao4_low[len(previsao4_low)-1])
    
    previsao1_close = float(previsao1_close[len(previsao1_close)-1])
    previsao2_close = float(previsao2_close[len(previsao2_close)-1])
    previsao3_close = float(previsao3_close[len(previsao3_close)-1])
    previsao4_close = float(previsao4_close[len(previsao4_close)-1])
    
    
    media_open = (previsao1_open+previsao2_open+previsao3_open+previsao4_open)/4
    
    media_high = (previsao1_high+previsao2_high+previsao3_high+previsao4_high)/4

    media_low = (previsao1_low+previsao2_low+previsao3_low+previsao4_low)/4

    media_close = (previsao1_close+previsao2_close+previsao3_close+previsao4_close)/4
    
    print("Previsão Open P1: ", previsao1_open, "P2: ",previsao2_open, "P3: ", previsao3_open, "P4: ", previsao4_open,"Média: ",media_open)
    print("Previsão High P1: ", previsao1_high, "P2: ",previsao2_high, "P3: ", previsao3_high, "P4: ", previsao4_high,"Média: ",media_high)
    print("Previsão Open P1: ", previsao1_low, "P2: ",previsao2_low, "P3: ", previsao3_low, "P4: ", previsao4_low,"Média: ",media_low)
    print("Previsão Close P1: ", previsao1_close, "P2: ",previsao2_close, "P3: ", previsao3_close, "P4: ", previsao4_close, "Média: ",media_close)  
    
    break


# ## Variáveis para o alerta
# preco_c é para colocar o valor mínimo previsto
# preco_v é colocar o valor máximo previsto
# 
# o api_key e api_secret é para colocar os dados do API da Binance
# 

# In[122]:


preco_c = 1600
preco_v = 1735
api_key = "colocar aqui a chave"
api_secret = "colocar aqui a chave"

client = Client(api_key, api_secret)


# ### Metodo de alertar quando valor de compra ou venda ultrapassar o valor esperado
# Infelizmente meu API não aceita compra e venda automaticamente, então tem que se colocar manualmente as compras e vendas no site da Binance

# In[ ]:


while(True):
    df=pd.DataFrame(client.get_all_tickers())
    df=df.set_index("symbol")
    df["price"]=df["price"].astype("float32")
    df.index=df.index.astype("string")
    preco = df.loc["ETHUSDT"]
    preco = preco.loc["price"]
    
    if preco <= preco_c:
        print("\n")
        print("Comprou!")
        print("\n")
        melodie = 'BBBB SSSS EEEE'
        for note in melodie:
            Beep(notes[note], 20)
            
    
    if preco >= preco_v:
        print("\n")
        print("Vendeu!")
        print("\n")
        melodie = 'AAAA FFFFF GGGG'
        for note in melodie:
            Beep(notes[note], 20)
    time.sleep(600)
            

