import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import pandas_ta as ta
import numpy as np
import requests
from bs4 import BeautifulSoup

# Function to fetch S&P 500 tickers from Wikipedia
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
    # '15m': timedelta(days=60),
    # '1h': timedelta(days=730),
    '1d': timedelta(days=3*365),
    # '1wk': timedelta(days=3*365)
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

# Applying the custom indicator and generating buy/sell signals
for tf in timeframes:
    for ticker in data[tf]:
        df = data[tf][ticker]
        
        # Round OHLC values to two decimal points
        df['Open'] = df['Open'].round(2)
        df['High'] = df['High'].round(2)
        df['Low'] = df['Low'].round(2)
        df['Close'] = df['Close'].round(2)
        
        # Linear Regression Curves
        df['reg1'] = ta.linreg(df['Close'], length=10)
        df['reg2'] = ta.linreg(df['Close'], length=14)
        df['reg3'] = ta.linreg(df['Close'], length=30)
        
        # R-squared Calculation
        r2_length = 14
        df['r2_raw'] = df['Close'].rolling(window=r2_length).apply(lambda x: np.corrcoef(x, np.arange(r2_length))[0, 1]**2)
        df['r2'] = df['r2_raw'] * 100  # Normalized to [0, 100]
        df['r2_smoothed'] = df['r2'].rolling(window=3).mean()
        
        # RSI Calculation
        df['rsi'] = ta.rsi(df['Close'], length=14)
        
        # Buy and Sell Signals
        df['buy_signal'] = np.where((df['r2_smoothed'] > 90) & (df['rsi'] < 30), 1, 0)
        df['sell_signal'] = np.where((df['r2_smoothed'] > 90) & (df['rsi'] > 70), 1, 0)
        # Filter rows with buy or sell signals
        df = df[(df['buy_signal'] == 1) | (df['sell_signal'] == 1)]
        
        data[tf][ticker] = df

# Save data to CSV files for each timeframe
for tf in timeframes:
    combined_data = pd.concat(data[tf], keys=data[tf].keys(), names=['Ticker', 'Date'])
    combined_data.to_csv(f'sp500_ohlc_data_{tf}.csv')

# Display a sample of the data
for tf in timeframes:
    print(f"Sample data for {tf}:")
    print(data[tf][tickers[0]].head())
