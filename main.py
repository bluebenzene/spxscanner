import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# Define the ticker symbols for S&P 500 and Small Caps
tickers = ['^GSPC', '^RUT']  # ^GSPC for S&P 500, ^RUT for Russell 2000 (representing small caps)

# Calculate the date range for the past 3 years
end_date = datetime.now()
start_date = end_date - timedelta(days=3*365)

# Download the historical data
data = yf.download(tickers, start=start_date, end=end_date, interval='1d')

# Display the first few rows of the data
print(data.head())

# Save the data to a CSV file
data.to_csv('sp500_and_small_caps_historical_data.csv')
