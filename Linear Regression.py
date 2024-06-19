import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.linear_model import LinearRegression

# Load and preprocess the data
data = pd.read_csv('USDINRX.csv')
data = data.dropna()
data = data.astype({'Date': 'datetime64[ns]'})

# Visualize the data
plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(data['Date'],data['Close'])
plt.xlabel('Date')
plt.ylabel('Close Price INR (₹)')
plt.show()

# Prepare the dataset
df = data.filter(['Close'])
dataset = df.values
train_data_len = math.ceil(len(dataset)*.8)

scaler = MinMaxScaler(feature_range=(0,1))
scaler_data = scaler.fit_transform(dataset)

train_data = scaler_data[0:train_data_len , :]
x_train = []
y_train = []

for i in range(60 , len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# Create and train the linear regression model
model = Sequential()
model.add(Dense(1, input_dim=60))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=1, batch_size=1)

# Print model summary
print(model.summary())

# Make predictions
test_data = scaler_data[train_data_len - 60:, :]
x_test = []

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)
pred = model.predict(x_test)
pred = scaler.inverse_transform(pred)

# Visualize the predictions
plt.figure(figsize=(16,8))
plt.plot(data['Date'][-len(pred):], pred, label='Predictions')
plt.plot(data['Date'][-len(pred):], data['Close'][-len(pred):], label='Actual')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Close Price INR (₹)')
plt.title('Actual vs Predicted Close Prices')
plt.show()


from sklearn.metrics import r2_score

r2 = r2_score(data['Close'][-len(pred):], pred)
print("R-squared (Coefficient of Determination):", r2)

