# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Data: PE Ratios and Dates
dates = [
    "2024-03-31", "2023-12-31", "2023-09-30", "2023-06-30", "2023-03-31", "2022-12-31",
    "2022-09-30", "2022-06-30", "2022-03-31", "2021-12-31", "2021-09-30", "2021-06-30",
    "2021-03-31", "2020-12-31", "2020-09-30", "2020-06-30", "2020-03-31", "2019-12-31",
    "2019-09-30", "2019-06-30", "2019-03-31", "2018-12-31", "2018-09-30", "2018-06-30",
    "2018-03-31", "2017-12-31", "2017-09-30", "2017-06-30", "2017-03-31", "2016-12-31",
    "2016-09-30", "2016-06-30", "2016-03-31", "2015-12-31", "2015-09-30", "2015-06-30",
    "2015-03-31", "2014-12-31", "2014-09-30", "2014-06-30", "2014-03-31", "2013-12-31",
    "2013-09-30", "2013-06-30", "2013-03-31", "2012-12-31", "2012-09-30", "2012-06-30",
    "2012-03-31", "2011-12-31", "2011-09-30", "2011-06-30", "2011-03-31", "2010-12-31",
    "2010-09-30", "2010-06-30", "2010-03-31", "2009-12-31"
]
pe_ratios = [
    None, 29.84, 27.80, 32.36, 27.75, 21.83, 22.35, 22.26, 27.93, 28.93, 24.74, 26.29, 
    26.85, 35.14, 34.68, 27.03, 19.38, 22.49, 18.27, 16.25, 15.32, 12.39, 18.10, 15.92, 
    15.30, 16.37, 15.71, 15.28, 15.63, 12.84, 12.59, 10.23, 11.04, 10.12, 10.78, 13.00,
    13.75, 13.25, 13.89, 13.25, 11.26, 12.14, 10.43, 8.53, 9.04, 10.27, 12.80, 11.58, 
    12.33, 9.73, 11.62, 11.21, 14.01, 15.19, 15.80, 17.21, 19.49, 20.52
]

# Ask user to input Q1 2024 PE ratio
pe_q1_2024 = float(input("Enter the PE Ratio for Q1 2024: "))
pe_ratios[0] = pe_q1_2024  # Replace the placeholder None with the provided value

# Create a DataFrame for organization
data = pd.DataFrame({'Date': dates, 'PE Ratio': pe_ratios})
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')

# Prepare features (X) and target (y) for time series regression
# We'll use the last 3 PE Ratios to predict the next one
window_size = 3
X = []
y = []

for i in range(len(data) - window_size):
    X.append(pe_ratios[i:i + window_size])  # Use a window of 3 PE Ratios
    y.append(pe_ratios[i + window_size])   # Predict the next PE Ratio

X = np.array(X)
y = np.array(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict Q2 2024 using the last 3 PE Ratios
next_input = np.array(pe_ratios[-window_size:]).reshape(1, -1)
next_pred = model.predict(next_input)

# Evaluate the model
mse = mean_squared_error(y_test, model.predict(X_test))
r2 = r2_score(y_test, model.predict(X_test))

print("\nModel Evaluation:")
print("Mean Squared Error (MSE):", mse)
print("R-squared (R2):", r2)

print("\nPredicted value for Q2 2024:")
print(f"Using the last 3 PE Ratios ({pe_ratios[-window_size:]}), the predicted PE Ratio for Q2 2024 is: {next_pred[0]:.2f}")

# Visualize the results
plt.figure(figsize=(10, 6))
plt.plot(range(len(y)), y, label='Actual PE Ratios', marker='o')
plt.plot(range(len(y)), model.predict(X), label='Predicted PE Ratios', linestyle='--', marker='x')
plt.title("PE Ratio Prediction")
plt.xlabel("Index")
plt.ylabel("PE Ratio")
plt.legend()
plt.show()
