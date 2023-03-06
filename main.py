import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


def train_test_split(data, split_ratio=0.8):
    split_index = int(len(data) * split_ratio)
    train_data, test_data = data[:split_index], data[split_index:]
    return train_data, test_data


def evaluate_model(predictions, actual):
    mse = np.mean((predictions - actual) ** 2)
    rmse = np.sqrt(mse)
    return rmse


def calculate_profit_loss(predictions, actual):
    profit_loss = 0
    for i in range(len(predictions)):
        if predictions[i] > actual[i]:
            profit_loss += predictions[i] - actual[i]
        else:
            profit_loss -= actual[i] - predictions[i]
    return profit_loss


def predict_next_day_price(data):
    train_data = data[:-1]
    test_data = data[-1:]
    train_ar = train_data['Close'].values
    history = [x for x in train_ar]

    # Build the model
    model = sm.tsa.ARIMA(history, order=(1, 0, 0))
    model_fit = model.fit()

    # Make prediction
    output = model_fit.forecast()
    yhat = output[0]

    return yhat


# Load the data
df = pd.read_csv('^NSEI.csv')

# Convert the 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Set the 'Date' column as index
df.set_index('Date', inplace=True)

# Split the data into train and test sets
train_data, test_data = train_test_split(df)

# Prepare the data for modeling
train_ar = train_data['Close'].values
test_ar = test_data['Close'].values
history = [x for x in train_ar]
predictions = list()

# Build the ARIMA model and make predictions on the test set
model = sm.tsa.ARIMA(history, order=(1, 0, 0))
model_fit = model.fit()
for t in range(len(test_ar)):
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test_ar[t]
    history.append(obs)
    model_fit = sm.tsa.ARIMA(history, order=(1, 0, 0)).fit()

# Evaluate the model
rmse = evaluate_model(predictions, test_ar)
print('ARIMA RMSE:', rmse)

# Calculate the profit/loss based on the ARIMA predictions
pl = calculate_profit_loss(predictions, test_ar)
print('ARIMA Profit/Loss:', pl)

# Plot the results
plt.plot(test_ar, label='Actual')
plt.plot(predictions, label='Predicted')
plt.legend()
plt.show()

# Call the function to get the live prediction
next_day_price = predict_next_day_price(df)

# Print the predicted price
print('Next day price:', next_day_price)
