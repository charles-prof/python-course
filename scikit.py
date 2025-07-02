import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Load the weather data
df = pd.read_csv('data/simple-weather.csv')

# Convert 'date' to datetime and then to ordinal for regression
df['date'] = pd.to_datetime(df['date'])
df['date_ordinal'] = df['date'].map(pd.Timestamp.toordinal)

# Prepare data for regression
X = df['date_ordinal'].values.reshape(-1, 1)
y = df['temperature_max'].values

# Fit linear regression
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Plot original data and regression line
plt.figure(figsize=(10, 6))
plt.scatter(df['date'], df['temperature_max'], color='blue', label='Max Temperature')
plt.plot(df['date'], y_pred, color='red', linewidth=2, label='Linear Regression')
plt.xlabel('Date')
plt.ylabel('Max Temperature')
plt.title('Max Temperature Over Time with Linear Regression')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
