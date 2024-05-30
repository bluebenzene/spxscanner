import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import pandas_ta as ta
import requests
from bs4 import BeautifulSoup
# Function to fetch S&P 500 tickers from Wikipedia
# Define the timeframes  Valid intervals: [1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo]

def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'id': 'constituents'})
    tickers = [row.find('td').text.strip() for row in table.find_all('tr')[1:]]
    return tickers

# Get the S&P 500 tickers
sp500_tickers = get_sp500_tickers()

# Save the tickers to a CSV file
pd.DataFrame(sp500_tickers, columns=["Ticker"]).to_csv('sp500_tickers.csv', index=False)

# Read the tickers from the CSV file
tickers_df = pd.read_csv('sp500_tickers.csv')
tickers = tickers_df['Ticker'].tolist()

# Define the timeframes and their corresponding date ranges
timeframes = {
    '15m': timedelta(days=60),
    '1h': timedelta(days=730),
    '1d': timedelta(days=3*365),
    '1wk': timedelta(days=3*365)
}

# Function to download data for a given ticker and timeframe
def download_data(ticker, timeframe, start_date, end_date):
    return yf.download(ticker, start=start_date, end=end_date, interval=timeframe)

# Dictionary to store data for each timeframe
data = {tf: {} for tf in timeframes}

# Download data for each ticker and timeframe
end_date = datetime.now()
for ticker in tickers:
    for tf, delta in timeframes.items():
        start_date = end_date - delta
        try:
            data[tf][ticker] = download_data(ticker, tf, start_date, end_date)
        except Exception as e:
            print(f"Failed to download data for {ticker} with timeframe {tf}: {e}")

# Applying technical indicators using pandas_ta
for tf in timeframes:
    for ticker in data[tf]:
        df = data[tf][ticker]
        df.ta.sma(length=10, append=True)  # Example: Simple Moving Average
        df.ta.rsi(length=14, append=True)  # Example: Relative Strength Index
        # Add more indicators as needed
        data[tf][ticker] = df

# Save data to CSV files for each timeframe
for tf in timeframes:
    combined_data = pd.concat(data[tf], keys=data[tf].keys(), names=['Ticker', 'Date'])
    combined_data.to_csv(f'sp500_ohlc_data_{tf}.csv')

# Display a sample of the data
for tf in timeframes:
    print(f"Sample data for {tf}:")
    print(data[tf][tickers[0]].head())
