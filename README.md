#import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
file_path = "/content/yahoo_crypto_data.csv"  
data = pd.read_csv(file_path)

# Display the first few rows
print(data.head())

data.describe() 

# Check if required columns exist
required_columns = ['Date', 'Close']
if not all(col in data.columns for col in required_columns):
    raise ValueError("Dataset must contain 'Date' and 'Close' columns.")

# Preprocess the data
data['Date'] = pd.to_datetime(data['Date'])
data.sort_values('Date', inplace=True)
data.set_index('Date', inplace=True)

#EDA
plt.figure(figsize=(15, 5))
plt.plot(data['High'])
plt.title('Bitcoin High price.', fontsize=15)
plt.ylabel('Price in dollars.')
plt.show()


plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
sns.histplot(data['Market Cap'], bins=20, kde=True)
plt.title('Distribution of Market Cap')

plt.subplot(1, 2, 2)
sns.histplot(data['High'], bins=20, kde=True)
plt.title('Distribution of High price ')

plt.tight_layout()
plt.show()

# Use only the 'Close' price for simplicity
prices = data['Close'].values.reshape(-1, 1)

# Scale data to [0, 1] range
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices)

# Create sequences for LSTM
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    return np.array(X), np.array(y)

sequence_length = 30  # Days to look back
X, y = create_sequences(scaled_prices, sequence_length)

# Split into training and testing sets
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32, verbose=1)

# Predict on test data
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)  # Reverse scaling
y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate metrics
mse = mean_squared_error(y_test_unscaled, predictions)
r2 = r2_score(y_test_unscaled, predictions)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R² Score: {r2 * 100:.2f}%")

# Plot actual vs predicted prices
plt.figure(figsize=(10, 6))
plt.plot(y_test_unscaled, label="Actual Prices", color="blue")
plt.plot(predictions, label="Predicted Prices", color="red", linestyle="dashed")
plt.legend()
plt.title("BTC Price Prediction using LSTM")
plt.xlabel("Test Data Points")
plt.ylabel("BTC Price")
plt.show()
from google.colab import files
import matplotlib.pyplot as plt
