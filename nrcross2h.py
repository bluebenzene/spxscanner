import os
import pytz
import requests
import pandas as pd
import pandas_ta as ta
import numpy as np
import yfinance as yf
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

############################
# 1. Time Check for Every 2 Hours
############################
def can_run_now():
    """
    Checks if at least 2 hours have passed since the last script execution.
    Stores the last run timestamp in an environment variable.
    """
    eastern = pytz.timezone('US/Eastern')
    now = datetime.now(eastern)
    
    last_run = os.getenv("LAST_RUN_TIME")
    
    if last_run:
        last_run_time = datetime.fromisoformat(last_run)
        if now - last_run_time < timedelta(hours=2):
            print("Less than 2 hours since last run. Exiting.")
            return False

    # Update last run time
    os.system(f'heroku config:set LAST_RUN_TIME="{now.isoformat()}"')
    return True

############################
# 2. Market Open Check
############################
def is_us_market_open():
    """
    Checks if the current time is within US market hours (9:30 AM - 4:00 PM ET)
    and not on weekends.
    """
    eastern = pytz.timezone('US/Eastern')
    now = datetime.now(eastern)
    
    if now.weekday() in [5, 6]:  # Saturday or Sunday
        return False
    
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    
    return market_open <= now <= market_close

############################
# 3. Telegram Functions
############################
def send_telegram_message(message: str):
    """
    Sends a given message to a Telegram chat.
    """
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    data = {"chat_id": chat_id, "text": message}
    requests.post(url, data=data)

############################
# 4. Utility Functions
############################
def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'id': 'constituents'})
    return [row.find('td').text.strip() for row in table.find_all('tr')[1:]]

def download_data(ticker, timeframe, start_date, end_date):
    return yf.download(ticker, start=start_date, end=end_date, interval=timeframe)

############################
# 5. Main Screener
############################
def main():
    if not can_run_now():
        return
    
    if not is_us_market_open():
        print("The US market is currently closed. Script execution halted.")
        return
    
    os.makedirs('stockdata', exist_ok=True)
    
    sp500_tickers = get_sp500_tickers()
    pd.DataFrame(sp500_tickers, columns=["Ticker"]).to_csv('stockdata/sp500_tickers.csv', index=False)
    tickers = pd.read_csv('stockdata/sp500_tickers.csv')['Ticker'].tolist()
    
    timeframes = {'1h': timedelta(days=30)}
    recent_period = 2  # lookback (hours) for new signals
    end_date = datetime.now()
    
    data = {tf: {} for tf in timeframes}
    
    for ticker in tickers:
        for tf, delta in timeframes.items():
            start_date = end_date - delta
            try:
                df = download_data(ticker, tf, start_date, end_date)
                if df.empty:
                    continue
                
                df.columns = df.columns.get_level_values(0) if isinstance(df.columns, pd.MultiIndex) else df.columns
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                if not all(col in df.columns for col in required_cols):
                    continue
                
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
    
    screener_results = []
    
    for tf in timeframes:
        for ticker, df in data[tf].items():
            df['reg1'] = ta.linreg(df['Close'], length=25)
            df['reg2'] = ta.linreg(df['Close'], length=50)
            
            df['buy_signal'] = np.where((df['reg1'] > df['reg2']) & (df['reg1'].shift(1) <= df['reg2'].shift(1)), 1, 0)
            df['sell_signal'] = np.where((df['reg1'] < df['reg2']) & (df['reg1'].shift(1) >= df['reg2'].shift(1)), 1, 0)
            
            df.to_csv(f'stockdata/{ticker}_{tf}_data.csv')
            
            recent_cutoff = df.index.max() - pd.Timedelta(hours=recent_period)
            signals_df = df[df.index >= recent_cutoff]
            signals_df = signals_df[(signals_df['buy_signal'] == 1) | (signals_df['sell_signal'] == 1)]
            
            for idx, row in signals_df.iterrows():
                screener_results.append({
                    'Ticker': ticker,
                    'Date': idx,
                    'Buy_Signal': row['buy_signal'],
                    'Sell_Signal': row['sell_signal']
                })
    
    screener_df = pd.DataFrame(screener_results)
    screener_df.to_csv('Regression_cross_screener_results_2h.csv', index=False)
    
    if not screener_df.empty:
        message = (
            "Regression Cross Screener (sent every 2 hrs)\n"
            "Buy = linreg(25) crosses above linreg(50)\n"
            "Sell = linreg(25) crosses below linreg(50)\n\n"
            + screener_df.to_string(index=False)
        )
        send_telegram_message(message)

############################
# 6. Entry Point
############################
if __name__ == "__main__":
    main()
