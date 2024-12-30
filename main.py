import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import pandas_ta as ta
import numpy as np
import requests
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv
import pytz  # Import pytz for timezone handling

load_dotenv()

# Function to fetch S&P 500 tickers from Wikipedia
def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'id': 'constituents'})
    tickers = [row.find('td').text.strip() for row in table.find_all('tr')[1:]]
    return tickers

# Function to send message to Telegram
def send_telegram_message(message):
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    data = {"chat_id": chat_id, "text": message}
    requests.post(url, data=data)

# Get the S&P 500 tickers
sp500_tickers = get_sp500_tickers()

# Save the tickers to a CSV file
pd.DataFrame(sp500_tickers, columns=["Ticker"]).to_csv('sp500_tickers.csv', index=False)

# Read the tickers from the CSV file
tickers_df = pd.read_csv('sp500_tickers.csv')
tickers = tickers_df['Ticker'].tolist()

# Define the timeframes and their corresponding date ranges
timeframes = {
    '1h': timedelta(days=30),
    '1d': timedelta(days=90)
}

# Restrict the script to run only during US market hours
us_timezone = pytz.timezone("America/New_York")
current_time = datetime.now(us_timezone)
market_open = current_time.replace(hour=9, minute=30, second=0, microsecond=0)
market_close = current_time.replace(hour=16, minute=0, second=0, microsecond=0)

if current_time < market_open or current_time > market_close:
    print("Market is closed. Exiting.")
    exit()

# Define the recent period (in days)
recent_period = 1

# Function to download data for a given ticker and timeframe
def download_data(ticker, timeframe, start_date, end_date):
    interval_map = {
        '1h': '60m',
        '1d': '1d'
    }
    return yf.download(ticker, start=start_date, end=end_date, interval=interval_map[timeframe])

# Dictionary to store data for each timeframe
data = {tf: {} for tf in timeframes}

# Download data for each ticker and timeframe
end_date = datetime.now(us_timezone)
for ticker in tickers:
    for tf, delta in timeframes.items():
        start_date = end_date - delta
        try:
            data[tf][ticker] = download_data(ticker, tf, start_date, end_date)
        except Exception as e:
            print(f"Failed to download data for {ticker} with timeframe {tf}: {e}")

# List to store screener results
screener_results = []

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
        r2_length = 25
        df['r2_raw'] = df['Close'].rolling(window=r2_length).apply(lambda x: np.corrcoef(x, np.arange(r2_length))[0, 1]**2)
        df['r2'] = df['r2_raw'] * 100  # Normalized to [0, 100]
        df['r2_smoothed'] = df['r2'].rolling(window=3).mean()
        
        # RSI Calculation
        df['rsi'] = ta.rsi(df['Close'], length=14)
        
        # Buy and Sell Signals
        df['buy_signal'] = np.where((df['r2_smoothed'] > 90) & (df['rsi'] < 30), 1, 0)
        df['sell_signal'] = np.where((df['r2_smoothed'] > 90) & (df['rsi'] > 70), 1, 0)
        
        # Filter for recent periods
        recent_date_cutoff = df.index.max() - pd.Timedelta(days=recent_period)
        df = df[df.index >= recent_date_cutoff]
        
        # Filter rows with buy or sell signals
        df_filtered = df[(df['buy_signal'] == 1) | (df['sell_signal'] == 1)]
        df_filtered['Timeframe'] = tf
        
        # Append results to screener list
        if not df_filtered.empty:
            for index, row in df_filtered.iterrows():
                screener_results.append({
                    'Ticker': ticker,
                    'Date': index,
                    'Buy Signal': row['buy_signal'],
                    'Sell Signal': row['sell_signal'],
                    'Timeframe': tf
                })

# Save screener results to a CSV file
screener_df = pd.DataFrame(screener_results)
screener_df.to_csv('screener_results.csv', index=False)

# Send results to Telegram
if not screener_df.empty:
    message = "Screener results:\n" + screener_df.to_string(index=False)
    send_telegram_message(message)

# Display a sample of the screener results
print("Screener results:")
print(screener_df.head())
