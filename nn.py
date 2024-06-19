import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from keras.layers import Dropout

data = pd.read_csv('USDINRX.csv')
data.head()
data.info()
data.isnull().sum()
data = data.dropna()
data.isnull().sum()
data = data.astype({'Date': 'datetime64[ns]'})
date = data['Date']
plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(data['Date'],data['Close'])
plt.xlabel('Date')
plt.ylabel('Close Price INR (₹)')
plt.show()
df = data.filter(['Close'])
dataset = df.values
train_data_len = math.ceil(len(dataset)*.8)
train_data_len
scaler = MinMaxScaler(feature_range=(0,1))
scaler_data = scaler.fit_transform(dataset)
scaler_data
train_data = scaler_data[0:train_data_len , :]
x_train = []
y_train = []

for i in range(60 , len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = Sequential()

model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1],1)))


model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
history=model.fit(x_train, y_train, epochs=10, batch_size=1)
test_data = scaler_data[train_data_len - 60:, :]
x_test = []
y_test = dataset[train_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
pred = model.predict(x_test)
pred = scaler.inverse_transform(pred)
train = data[:train_data_len]
valid = data[train_data_len:]
valid['Predictions'] = pred
plt.figure(figsize=(16,8))
plt.xlabel('Date')
plt.ylabel('Close Price INR (₹)')
plt.plot(train['Date'],train['Close'])
plt.plot(valid['Date'],valid[['Close','Predictions']])
plt.legend(['Train','Valid','Predictions'], loc='lower right')
plt.show()
plt.figure(figsize=(16,8))
plt.plot(valid['Date'],valid[['Close','Predictions']])
plt.legend(['Valid','Predictions'], loc='lower right')
plt.xlabel('Date')
plt.ylabel('Close Price INR (₹)')
plt.show()


# Evaluate model on test data
test_predictions = model.predict(x_test)
test_predictions = scaler.inverse_transform(test_predictions)

# Calculate MSE
mse = np.mean(np.square(test_predictions - y_test))
print("Mean Squared Error (MSE):", mse)

# Plot MSE
plt.figure(figsize=(16, 8))
plt.plot(history.history['loss'], label='Training Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()

from sklearn.metrics import r2_score

# Calculate R-squared for test set
test_r2 = r2_score(y_test, test_predictions)
r2_percent = test_r2 * 100
print("Accuracy:", r2_percent, "%")




