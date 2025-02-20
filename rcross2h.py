import os
import pytz
import requests
import pandas as pd
import pandas_ta as ta
import numpy as np
import yfinance as yf
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import time
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
    
    # Market closed on weekends
    if now.weekday() in [5, 6]:  # Saturday (5) or Sunday (6)
        return False
    
    market_open = now.replace(hour=9, minute=25, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=15, second=0, microsecond=0)
    
    return market_open <= now <= market_close


############################
# 2. Telegram Functions
############################
def send_telegram_message(message: str):
    """
    Sends a given message to a Telegram chat.
    Fill in your own bot token and chat ID below.
    """
    bot_token = "YOUR_TELEGRAM_BOT_TOKEN"
    chat_id   = "YOUR_CHAT_ID"

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    data = {"chat_id": chat_id, "text": message}
    response = requests.post(url, data=data)
    return response.json()

def can_send_telegram_message(file_path='telegram_time.txt', threshold_hours=2):
    """
    Checks if at least `threshold_hours` have passed since last Telegram message.
    Uses a file for persistence between script runs.
    Note: On Heroku, the file system is ephemeral across dyno restarts.
    """
    eastern = pytz.timezone('US/Eastern')
    now = datetime.now(eastern)

    if not os.path.exists(file_path):
        # No file -> no previous send time -> we can send now
        return True
    
    with open(file_path, 'r') as f:
        timestamp_str = f.read().strip()
        if not timestamp_str:
            # File was empty
            return True
        
        # Parse the stored ISO-format datetime
        last_sent_time = datetime.fromisoformat(timestamp_str)

    # Compare now vs. last send time
    return (now - last_sent_time) >= timedelta(hours=threshold_hours)

def update_telegram_send_time(file_path='telegram_time.txt'):
    """
    Writes the current time to a file, marking the last Telegram send time.
    """
    eastern = pytz.timezone('US/Eastern')
    now = datetime.now(eastern)
    with open(file_path, 'w') as f:
        f.write(now.isoformat())


############################
# 3. Utility Functions
############################
def get_sp500_tickers():
    """
    Fetches S&P 500 tickers from Wikipedia.
    """
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'id': 'constituents'})
    tickers = [row.find('td').text.strip() for row in table.find_all('tr')[1:]]
    return tickers

def download_data(ticker, timeframe, start_date, end_date):
    """
    Wrapper around yfinance to download the specified interval data.
    """
    return yf.download(ticker, start=start_date, end=end_date, interval=timeframe)


############################
# 4. Main Screener
############################
def main():
    # 4A. Check if the market is open. If not, exit.
    if not is_us_market_open():
        print("The US market is currently closed. Script execution halted.")
        return

    # 4B. Prepare directories
    os.makedirs('stockdata', exist_ok=True)

    # 4C. Get and store S&P 500 tickers (caching them to CSV for reference)
    sp500_tickers = get_sp500_tickers()
    pd.DataFrame(sp500_tickers, columns=["Ticker"]) \
        .to_csv('stockdata/sp500_tickers.csv', index=False)

    # 4D. Read them back from CSV
    tickers_df = pd.read_csv('stockdata/sp500_tickers.csv')
    tickers = tickers_df['Ticker'].tolist()

    # 4E. Define timeframe & range
    timeframes = {'1h': timedelta(days=30)}
    recent_period = 2  # lookback (hours) for new signals
    end_date = datetime.now()

    # 4F. Dictionary to store data
    data = {tf: {} for tf in timeframes}

    # 4G. Download & resample data
    for ticker in tickers:
        for tf, delta in timeframes.items():
            start_date = end_date - delta
            try:
                df = download_data(ticker, tf, start_date, end_date)
                time.sleep(1)
                if df.empty:
                    print(f"No data for {ticker} on {tf}. Skipping.")
                    continue
                
                # Flatten MultiIndex columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                # Check required columns
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                if not all(col in df.columns for col in required_cols):
                    print(f"Missing columns in {ticker} data. Skipping.")
                    continue

                # Resample 1-hour to 2-hour bars
                df_resampled = df.resample('2h').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }).dropna()
                
                data[tf][ticker] = df_resampled
            except Exception as e:
                print(f"Failed downloading {ticker} on {tf}: {e}")

    # 4H. List to store screening results
    screener_results = []

    # 4I. Loop over data, compute signals
    for tf in timeframes:
        for ticker, df in data[tf].items():
            # Round OHLC
            df['Open']  = df['Open'].round(2)
            df['High']  = df['High'].round(2)
            df['Low']   = df['Low'].round(2)
            df['Close'] = df['Close'].round(2)
            
            # LinReg(25) & LinReg(50)
            df['reg1'] = ta.linreg(df['Close'], length=25)
            df['reg2'] = ta.linreg(df['Close'], length=50)
            
            # Buy signals: reg1 crosses above reg2
            df['buy_signal'] = np.where(
                (df['reg1'] > df['reg2']) & (df['reg1'].shift(1) <= df['reg2'].shift(1)),
                1, 0
            )

            # Sell signals: reg1 crosses below reg2
            df['sell_signal'] = np.where(
                (df['reg1'] < df['reg2']) & (df['reg1'].shift(1) >= df['reg2'].shift(1)),
                1, 0
            )

            # Save CSV
            df.to_csv(f'stockdata/{ticker}_{tf}_data.csv')

            # Filter to last 'recent_period' hours
            if not df.empty:
                recent_cutoff = df.index.max() - pd.Timedelta(hours=recent_period)
                df_recent = df[df.index >= recent_cutoff]
                
                # Check signals
                signals_df = df_recent[(df_recent['buy_signal'] == 1) | (df_recent['sell_signal'] == 1)]
                if not signals_df.empty:
                    for idx, row in signals_df.iterrows():
                        screener_results.append({
                            'Ticker': ticker,
                            'Date': idx,
                            'Buy_Signal': row['buy_signal'],
                            'Sell_Signal': row['sell_signal']
                        })

    # 4J. Convert results to DataFrame and CSV
    screener_df = pd.DataFrame(screener_results)
    screener_df.to_csv('Regression_cross_screener_results_2h.csv', index=False)

    # 4K. Print sample
    print("Screener Results (Hourly run, 2hr signals window):")
    if not screener_df.empty:
        print(screener_df.tail())
    else:
        print("No buy/sell signals found in the last 2 hours.")

    ############################
    # 5. Send to Telegram every 2 hours
    ############################
    if not screener_df.empty and can_send_telegram_message():
        message = (
            "Regression Cross Screener (sent every 2 hrs)\n"
            "Buy = linreg(25) crosses above linreg(50)\n"
            "Sell = linreg(25) crosses below linreg(50)\n\n"
            + screener_df.to_string(index=False)
        )
        send_telegram_message(message)
        update_telegram_send_time()


############################
# 6. Entry Point
############################
if __name__ == "__main__":
    main()
