import pandas as pd
import matplotlib.pyplot as plt

# Load the weather data
df = pd.read_csv('data/simple-weather.csv')

# Convert 'date' to datetime
df['date'] = pd.to_datetime(df['date'])

plt.figure(figsize=(10, 6))
plt.plot(df['date'], df['temperature_max'], marker='o', linestyle='-', label='Max Temperature')
plt.plot(df['date'], df['temperature_min'], marker='o', linestyle='-', label='Min Temperature')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Temperature Max and Min Over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
