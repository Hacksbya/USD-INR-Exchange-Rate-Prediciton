import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Load and preprocess the data
data = pd.read_csv('USDINRX.csv').dropna()
data['Date'] = pd.to_datetime(data['Date'])

# Print information about the data
print("Data Information:")
print(data.info())

# Visualize the data
plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(data['Date'], data['Close'])
plt.xlabel('Date')
plt.ylabel('Close Price INR (₹)')
plt.show()

# Prepare the dataset
df = data.filter(['Close'])
dataset = df.values
scaler = MinMaxScaler(feature_range=(0,1))
scaler_data = scaler.fit_transform(dataset)

# Split the data into train and test sets
train_size = int(len(scaler_data) * 0.8)
test_size = len(scaler_data) - train_size
train_data, test_data = scaler_data[0:train_size,:], scaler_data[train_size:len(scaler_data),:]

def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data)-time_step-1):
        a = data[i:(i+time_step), 0]
        X.append(a)
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 60
x_train, y_train = create_dataset(train_data, time_step)
x_test, y_test = create_dataset(test_data, time_step)

# Ridge Regression
ridge = Ridge(alpha=0.1)
ridge.fit(x_train, y_train)

# Make predictions
train_predict = ridge.predict(x_train)
test_predict = ridge.predict(x_test)

# Invert predictions
train_predict = scaler.inverse_transform(train_predict.reshape(-1,1))
y_train = scaler.inverse_transform(y_train.reshape(-1,1))
test_predict = scaler.inverse_transform(test_predict.reshape(-1,1))
y_test = scaler.inverse_transform(y_test.reshape(-1,1))

# Calculate RMSE
train_rmse = np.sqrt(mean_squared_error(y_train, train_predict))
test_rmse = np.sqrt(mean_squared_error(y_test, test_predict))
print("\nRMSE (Root Mean Squared Error):")
print("Train RMSE:", train_rmse)
print("Test RMSE:", test_rmse)

# Shift the date array by the time step
date_test_shifted = data['Date'][2*time_step+1:]

# Match the length of date_test_shifted with test_predict
date_test_shifted = date_test_shifted[:len(test_predict)]

# Visualize the predictions
plt.figure(figsize=(16,8))
plt.plot(date_test_shifted, y_test, label='Actual (Test)')
plt.plot(date_test_shifted, test_predict, label='Predicted (Test)')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Close Price INR (₹)')
plt.title('Actual vs Predicted Close Prices using Ridge Regression')
plt.show()
