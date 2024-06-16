import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing dataset
data = pd.read_csv('Google_Stock_Price_Train.csv')
train_data = data.iloc[:,1:2].values

#Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
train_data_scaled = sc.fit_transform(train_data)

#Making data structure with some timestamps t
t = 60
X_train = []
y_train = [] 
for i in range(t,train_data_scaled.shape[0]):
    temp=[]
    for j in range(i-60,i):
        temp.append(train_data_scaled[j][0])
    X_train.append(temp)
    y_train.append(train_data_scaled[i][0])
X_train = np.array(X_train)
y_train = np.array(y_train)

#Reshaping our input
X_train = np.reshape(X_train , (X_train.shape[0], X_train.shape[1], 1))

#Creating LSTM layers
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout

#rnn = Sequential()

#rnn.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
#rnn.add(Dropout(0.2))
#rnn.add(LSTM(units=50, return_sequences=True))
#rnn.add(Dropout(0.2))
#rnn.add(LSTM(units=50, return_sequences=True))
#rnn.add(Dropout(0.2))
#rnn.add(LSTM(units=50))
#rnn.add(Dropout(0.2))
#rnn.add(Dense(units=1))

#rnn.compile(optimizer='adam', loss='mean_squared_error')

#rnn.fit(X_train, y_train, epochs=25, batch_size=32)


















    


