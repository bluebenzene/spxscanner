import os
import pytz
import requests
import pandas as pd
import pandas_ta as ta
import numpy as np
import yfinance as yf
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv('/home/ubuntu/spxscanner/.env')

############################
# 1. Market Open Check
############################
def is_us_market_open():
    """
    Checks if the current time is within US market hours (9:30 AM - 4:00 PM ET)
    and not on weekends.
    """
    eastern = pytz.timezone('US/Eastern')
    now = datetime.now(eastern)
    if now.weekday() in [5, 6]:  # Saturday and Sunday
        return False
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= now <= market_close

############################
# 2. Telegram Functions
############################
def send_telegram_message(message):
    """
    Sends a given message to a Telegram chat.
    """
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    data = {"chat_id": chat_id, "text": message}
    requests.post(url, data=data)

############################
# 3. Utility Functions
############################
def get_sp500_tickers():
    """
    Fetches the S&P 500 tickers from Wikipedia.
    """
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'id': 'constituents'})
    return [row.find('td').text.strip() for row in table.find_all('tr')[1:]]

# Note: This download_data function is no longer used since we use a bulk download.
# def download_data(ticker, timeframe, start_date, end_date):
#     return yf.download(ticker, start=start_date, end=end_date, interval=timeframe)

############################
# 4. Bulk Download Setup using CachedLimiterSession and yf.Tickers
############################
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

############################
# 5. Main Screener
############################
def main():
    if not is_us_market_open():
        print("The US market is currently closed. Script execution halted.")
        return
    
    # Create directory to store data
    os.makedirs('stockdata', exist_ok=True)
    
    # Get S&P 500 tickers and save to CSV
    sp500_tickers = get_sp500_tickers()
    pd.DataFrame(sp500_tickers, columns=["Ticker"]).to_csv('stockdata/sp500_tickers.csv', index=False)
    
    # Read tickers from CSV
    tickers = pd.read_csv('stockdata/sp500_tickers.csv')['Ticker'].tolist()
    
    # Define the timeframe for bulk downloading 60m data over the past 30 days.
    # We will later resample these bars to 2-hour bars.
    timeframes = {
        '60m': timedelta(days=30),
    }
    recent_period = 2  # Look back the last 2 hours for new signals
    # Bulk download period string based on days
    # (e.g., '30d' for 30 days)
    
    # Create a Tickers object with the custom CachedLimiterSession
    tickers_str = " ".join(tickers)
    dat = yf.Tickers(tickers_str, session=session)
    
    data = {}
    screener_results = []  # list to store screener signal results
    
    # Loop over defined timeframes (in this case, only '60m')
    for tf, delta in timeframes.items():
        period_str = f"{delta.days}d"
        try:
            his_data = dat.history(period=period_str, interval=tf)
        except Exception as e:
            print(f"Error downloading bulk data for interval {tf}: {e}")
            continue
        
        data[tf] = {}
        # Extract available tickers from the MultiIndex level 'Ticker'
        available_tickers = set(his_data.columns.get_level_values('Ticker'))
        for ticker in tickers:
            if ticker not in available_tickers:
                print(f"Ticker {ticker} not found in historical data, skipping.")
                continue
            try:
                # Extract data for a single ticker and make a copy
                df = his_data.xs(ticker, axis=1, level='Ticker').copy()
                # Flatten the columns (Price types) if needed
                df.columns = df.columns.get_level_values(0)
                # Round OHLC values to two decimals to avoid floating point issues
                for col in ['Open', 'High', 'Low', 'Close']:
                    if col in df.columns:
                        df.loc[:, col] = df[col].round(2)
                
                # Resample the 60m data into 2-hour bars
                df_resampled = df.resample('2h').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }).dropna()
                
                # Calculate linear regression curves using pandas_ta
                df_resampled['reg1'] = ta.linreg(df_resampled['Close'], length=25)
                df_resampled['reg2'] = ta.linreg(df_resampled['Close'], length=50)
                
                # Generate buy and sell signals
                df_resampled['buy_signal'] = np.where(
                    (df_resampled['reg1'] > df_resampled['reg2']) & (df_resampled['reg1'].shift(1) <= df_resampled['reg2'].shift(1)),
                    1, 0
                )
                df_resampled['sell_signal'] = np.where(
                    (df_resampled['reg1'] < df_resampled['reg2']) & (df_resampled['reg1'].shift(1) >= df_resampled['reg2'].shift(1)),
                    1, 0
                )
                
                # Save the resampled data to a CSV file for this ticker
                csv_filename = f'stockdata/{ticker}_2h_data.csv'
                df_resampled.to_csv(csv_filename)
                
                # Filter for the recent period (last 2 hours)
                if not df_resampled.empty:
                    recent_date_cutoff = df_resampled.index.max() - pd.Timedelta(hours=recent_period)
                    df_recent = df_resampled[df_resampled.index >= recent_date_cutoff]
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
    screener_df.to_csv('Regression_cross_screener_results_2h.csv', index=False)
    
    # If signals exist, print a sample and send them via Telegram
    if not screener_df.empty:
        print(f"2hr Screener Results:\n{str(screener_df.tail())}")
        message = (
            "2hr Screener Results:\n"
            "Buy = linreg(25) crosses above linreg(50)\n"
            "Sell = linreg(25) crosses below linreg(50)\n\n"
            + screener_df.to_string(index=False)
        )
        send_telegram_message(message)
    else:
        print("No buy/sell signals found in the last 2 hours.")

############################
# 6. Entry Point
############################
if __name__ == "__main__":
    main()
