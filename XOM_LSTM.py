
#Import the libraries
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras import optimizers
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')



dataset = pd.read_csv('XOM Historical Data_10yrs.csv')
dataset = dataset[::-1]
from datetime import datetime
dataset['Date']=pd.to_datetime(dataset_train['Date'])
dataset.reset_index(drop=True, inplace=True)
training_set = dataset['Price'][0:int(0.8*len(dataset))]
test_set = dataset['Price'][int(0.8*len(dataset)):len(dataset)]
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(pd.DataFrame(training_set))
training_set_scaled.shape
X_train = []
y_train = []
for i in range(60, len(training_set)):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

regressor = Sequential()

regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

regressor.fit(X_train, y_train, epochs = 10, batch_size = 32)



#Prediction on test set
real_stock_price = test_set
dataset_total = pd.concat((training_set, test_set), axis = 0)
inputs = dataset_total[len(dataset_total) - len(test_set) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, len(test_set)+60):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


plt.plot(np.array(real_stock_price), color = 'black', label = 'XOM Stock Price')
plt.plot(predicted_stock_price, color = 'green', label = 'Predicted XOM Stock Price')
plt.title('XOM Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('XOM Stock Price')
plt.legend()
plt.show()


#Prediction of future dates
dataset_total = pd.concat((training_set, test_set), axis = 0)
inputs = dataset_total[len(dataset_total) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
predict_set = []

number_of_days = 90   #Number of days you want to pedict in future
predicted = []    #the predicted values in the future

for i in range(60,number_of_days+60):
    predict_set=inputs[i-60:i,0]
    predict_set = np.array(predict_set)
    predict_set = np.reshape(predict_set, (1, 60, 1))
    predicted_stock_price = regressor.predict(predict_set)
    predicted.append((predicted_stock_price))
    inputs=list(inputs)
    inputs.append(list(predicted_stock_price))
    inputs=np.array(inputs)

predicted = np.array(predicted)
predicted = predicted.reshape(predicted.shape[0],1)
predicted = sc.inverse_transform(predicted)

plt.plot(predicted, color = 'green', label = 'Predicted XOM Stock Price')
plt.title('XOM Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('XOM Stock Price')
plt.legend()
plt.show()




