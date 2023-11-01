import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import warnings
warnings.filterwarnings('ignore')

"""
I selected 1st use case of stock price prediction for Tesla (TSLA). Downloaded updated data from Yahoo Finance.
I have decided to use Linear Regression model for this use case, because I have only one independent variable (date) as 
input specified in the assignment. I have also only one dependent variable (close price). Linear Regression is good for
predicting continuous values, which is the case for stock price prediction.
I have split the data into training and testing sets. I have trained the model and predicted the price for the 
input date. I have also calculated the price and score if date is in the dataset, else just print the predicted price.
I have plotted the data and the prediction.

"""

# Load the data
data = pd.read_csv('TSLA.csv')  # https://finance.yahoo.com/quote/TSLA/history?p=TSLA
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True, format='%Y-%m-%d')
data['Days'] = (data['Date'] - data['Date'].min()).dt.days

# Date as independent variable, Close as dependent variable
X = data[['Days']]
y = data['Close']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)


# Helper functions
def prediction_percentage_score(predicted, true):
    return (1 - abs(predicted - true) / true) * 100


def predict_price(date, model, data):
    date = pd.to_datetime(date, dayfirst=True, format='%Y-%m-%d')
    days = (date - data['Date'].min()).days
    return model.predict([[days]])[0]


# Predict the price for the input date
date_to_predict = '2023-10-16'
predicted_price = predict_price(date_to_predict, model, data)

# Calculate the price and score if date is in the dataset, else just print the predicted price
if date_to_predict in data['Date'].astype(str).values:
    true_price = data[data['Date'] == date_to_predict]['Close'].values[0]
    score = prediction_percentage_score(predicted_price, true_price)
    print(f"Predicted price for {date_to_predict}: ${predicted_price:.2f}")
    print(f"True price for {date_to_predict}: ${true_price:.2f}")
    print(f"Prediction Percentage Score for {date_to_predict}: {score:.2f}%")
else:
    print(f"Predicted price for {date_to_predict}: ${predicted_price:.2f}")

# Plot the data and the prediction
import matplotlib.pyplot as plt

plt.figure(figsize=(16, 8))
plt.title('Tesla Stock Price')
plt.xlabel('Days')
plt.ylabel('Close Price USD ($)')
plt.plot(data['Days'], data['Close'])
plt.plot(X_test, model.predict(X_test), color='r')
plt.show()
