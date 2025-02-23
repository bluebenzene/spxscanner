import os
import time
import pytz
import requests
import pandas as pd
import numpy as np
import pandas_ta as ta
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import yfinance as yf

# Check if US market is open
def is_us_market_open():
    """Checks if the current time is within US market hours (9:30 AM - 4:00 PM ET) and not on weekends."""
    eastern = pytz.timezone('US/Eastern')
    now = datetime.now(eastern)
    if now.weekday() in [5, 6]:  # Saturday and Sunday
        return False
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= now <= market_close

# For debugging in Jupyter, you can disable the market check.
# if not is_us_market_open():
#     print("The US market is currently closed. Script execution halted.")
# else:
load_dotenv()

# Function to fetch S&P 500 tickers from Wikipedia
def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'id': 'constituents'})
    tickers = [row.find('td').text.strip() for row in table.find_all('tr')[1:]]
    return tickers

# Function to send a message via Telegram
def send_telegram_message(message):
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    data = {"chat_id": chat_id, "text": message}
    requests.post(url, data=data)

# Get the S&P 500 tickers and save to CSV
sp500_tickers = get_sp500_tickers()
os.makedirs('stockdata', exist_ok=True)
pd.DataFrame(sp500_tickers, columns=["Ticker"]).to_csv('stockdata/sp500_tickers.csv', index=False)

# Read tickers from CSV
tickers_df = pd.read_csv('stockdata/sp500_tickers.csv')
tickers = tickers_df['Ticker'].tolist()

# Define the timeframes (here using only daily data for 90 days)
timeframes = {
    '1d': timedelta(days=90),
}
recent_period = 1  # in days

# --- New download approach using CachedLimiterSession and yf.Tickers ---
from requests import Session
from requests_cache import CacheMixin, SQLiteCache
from requests_ratelimiter import LimiterMixin, MemoryQueueBucket
from pyrate_limiter import Duration, RequestRate, Limiter

class CachedLimiterSession(CacheMixin, LimiterMixin, Session):
    pass

session = CachedLimiterSession(
    limiter=Limiter(RequestRate(2, Duration.SECOND * 5)),  # max 2 requests per 5 seconds
    bucket_class=MemoryQueueBucket,
    backend=SQLiteCache("yfinance.cache"),
)

# Create a Tickers object using all tickers with the custom session.
# Note: yf.Tickers accepts a string of tickers separated by spaces.
tickers_str = " ".join(tickers)
dat = yf.Tickers(tickers_str, session=session)

data = {}
screener_results = []  # list to store screener signal results

for tf, delta in timeframes.items():
    # Use period string based on the desired date range, e.g. "90d" for 90 days
    period_str = f"{delta.days}d"
    try:
        his_data = dat.history(period=period_str, interval=tf)
    except Exception as e:
        print(f"Error downloading bulk data for interval {tf}: {e}")
        continue

    data[tf] = {}
    # Get available tickers in the downloaded data from the MultiIndex level 'Ticker'
    available_tickers = set(his_data.columns.get_level_values('Ticker'))
    for ticker in tickers:
        if ticker not in available_tickers:
            print(f"Ticker {ticker} not found in historical data, skipping.")
            continue
        try:
            # Extract data for a single ticker from the MultiIndex DataFrame and make a copy
            df = his_data.xs(ticker, axis=1, level='Ticker').copy()
            # Flatten the columns (Price types) if needed
            df.columns = df.columns.get_level_values(0)
            # Round OHLC values to two decimals using .loc to avoid SettingWithCopyWarning
            for col in ['Open', 'High', 'Low', 'Close']:
                if col in df.columns:
                    df.loc[:, col] = df[col].round(2)

            # Calculate linear regression curves using pandas_ta
            df['reg1'] = ta.linreg(df['Close'], length=25)
            df['reg2'] = ta.linreg(df['Close'], length=50)

            # Generate buy and sell signals
            df['buy_signal'] = np.where((df['reg1'] > df['reg2']) & (df['reg1'].shift(1) <= df['reg2'].shift(1)), 1, 0)
            df['sell_signal'] = np.where((df['reg1'] < df['reg2']) & (df['reg1'].shift(1) >= df['reg2'].shift(1)), 1, 0)

            # Save each ticker's data to a CSV file
            csv_filename = f'stockdata/{ticker}_{tf}_data.csv'
            df.to_csv(csv_filename)

            # Filter for the recent period
            if not df.empty:
                recent_date_cutoff = df.index.max() - pd.Timedelta(days=recent_period)
                df_recent = df[df.index >= recent_date_cutoff]
                df_filtered = df_recent[(df_recent['buy_signal'] == 1) | (df_recent['sell_signal'] == 1)]
                if not df_filtered.empty:
                    for index, row in df_filtered.iterrows():
                        screener_results.append({
                            'Ticker': ticker,
                            'Date': index,
                            'Buy Signal': row['buy_signal'],
                            'Sell Signal': row['sell_signal']
                        })
        except Exception as e:
            print(f"Failed to process data for ticker {ticker} on timeframe {tf}: {e}")

# Save the screener results to a CSV file
screener_df = pd.DataFrame(screener_results)
screener_df.to_csv('Regression_cross_screener_results_1d.csv', index=False)

# Optionally, send results to Telegram if available
if not screener_df.empty:
    message = (
        "Daily Screener Results:\n"
        "Buy = linreg 25 crossover linreg 50\n"
        "Sell = linreg 25 cross below linreg 50\n"
        f"{screener_df.to_string(index=False)}"
    )
    # Uncomment the following line to send the message:
    send_telegram_message(message)

# Display a sample of the screener results
print("Daily Screener Results:\nBuy = linreg 25 crossover linreg 50\nSell = linreg 25 cross below linreg 50\n")
print(screener_df.tail())
