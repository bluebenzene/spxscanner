import os
import time
import pytz
import requests
import pandas as pd
import pandas_ta as ta
import numpy as np
import yfinance as yf
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv('/home/ubuntu/spxscanner/.env')

############################
# Setup Logging
############################
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'screener.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

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
    Sends a given message using the default Telegram bot.
    """
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    data = {"chat_id": chat_id, "text": message}
    response = requests.post(url, data=data)
    logging.info(f"Sent message to default channel. Response status: {response.status_code}")
    return response

def send_cross_telegram_message(message):
    """
    Sends a given message using the cross Telegram bot.
    """
    cross_bot_token = os.getenv("CROSS_TELEGRAM_BOT_TOKEN")
    cross_chat_id = os.getenv("CROSS_TELEGRAM_CHAT_ID")
    url = f"https://api.telegram.org/bot{cross_bot_token}/sendMessage"
    data = {"chat_id": cross_chat_id, "text": message}
    response = requests.post(url, data=data)
    logging.info(f"Sent message to cross channel. Response status: {response.status_code}")
    return response

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
    tickers = [row.find('td').text.strip() for row in table.find_all('tr')[1:]]
    logging.info("Fetched S&P 500 tickers")
    return tickers

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
    # if not is_us_market_open():
    #     logging.info("The US market is currently closed. Script execution halted.")
    #     return
    
    # Create directory to store data
    os.makedirs('stockdata', exist_ok=True)
    
    # Get S&P 500 tickers and save to CSV
    sp500_tickers = get_sp500_tickers()
    pd.DataFrame(sp500_tickers, columns=["Ticker"]).to_csv('stockdata/sp500_tickers.csv', index=False)
    
    # Read tickers from CSV
    tickers = pd.read_csv('stockdata/sp500_tickers.csv')['Ticker'].tolist()
    
    # Define the timeframe for bulk downloading 60m data over the past 30 days.
    # We will later resample these bars to 2-hour bars.
    timeframes = {'60m': timedelta(days=30)}
    recent_period = 2  # Look back the last 2 hours for new signals
    
    # Create a Tickers object with the custom CachedLimiterSession
    tickers_str = " ".join(tickers)
    dat = yf.Tickers(tickers_str, session=session)
    
    # Containers for both screener results
    linreg_results = []   # Linear Regression signals
    r2_results = []       # R² indicator cross signals
    
    # Loop over defined timeframes (only '60m' here)
    for tf, delta in timeframes.items():
        period_str = f"{delta.days}d"
        try:
            his_data = dat.history(period=period_str, interval=tf)
            logging.info(f"Downloaded historical data for interval {tf}")
        except Exception as e:
            logging.error(f"Error downloading bulk data for interval {tf}: {e}")
            continue
        
        # Extract available tickers from the MultiIndex level 'Ticker'
        available_tickers = set(his_data.columns.get_level_values('Ticker'))
        for ticker in tickers:
            if ticker not in available_tickers:
                logging.warning(f"Ticker {ticker} not found in historical data, skipping.")
                continue
            try:
                # Extract data for a single ticker and make a copy
                df = his_data.xs(ticker, axis=1, level='Ticker').copy()
                # Flatten the columns if needed
                df.columns = df.columns.get_level_values(0)
                # Round OHLC values to two decimals
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
                
                # ---------------------------
                # Linear Regression Signals
                # ---------------------------
                df_resampled['reg1'] = ta.linreg(df_resampled['Close'], length=25)
                df_resampled['reg2'] = ta.linreg(df_resampled['Close'], length=50)
                df_resampled['buy_signal'] = np.where(
                    (df_resampled['reg1'] > df_resampled['reg2']) &
                    (df_resampled['reg1'].shift(1) <= df_resampled['reg2'].shift(1)),
                    1, 0
                )
                df_resampled['sell_signal'] = np.where(
                    (df_resampled['reg1'] < df_resampled['reg2']) &
                    (df_resampled['reg1'].shift(1) >= df_resampled['reg2'].shift(1)),
                    1, 0
                )
                
                # ---------------------------
                # R² Indicator Signals
                # ---------------------------
                # Compute hl2 as average of High and Low
                df_resampled['hl2'] = (df_resampled['High'] + df_resampled['Low']) / 2
                length = 25
                avg_len = 3
                threshold = 0.9

                def calc_r2(series):
                    if len(series) < length:
                        return np.nan
                    x = np.arange(length)
                    r = np.corrcoef(x, series)[0, 1]
                    return r**2

                df_resampled['r2'] = df_resampled['hl2'].rolling(window=length).apply(calc_r2, raw=False)
                df_resampled['r2_smoothed'] = df_resampled['r2'].rolling(window=avg_len).mean()
                df_resampled['cross_signal'] = np.where(
                    (df_resampled['r2_smoothed'].shift(1) > threshold) &
                    (df_resampled['r2_smoothed'] <= threshold),
                    1, 0
                )
                
                # Save the resampled data to CSV for reference
                csv_filename = f'stockdata/{ticker}_2h_data.csv'
                df_resampled.to_csv(csv_filename)
                logging.info(f"Processed and saved data for ticker {ticker}")
                
                # ---------------------------
                # Filter for Recent Signals (Last 2 Hours)
                # ---------------------------
                if not df_resampled.empty:
                    recent_cutoff = df_resampled.index.max() - pd.Timedelta(hours=recent_period)
                    df_recent = df_resampled[df_resampled.index >= recent_cutoff]
                    
                    # Linear regression signals filtering
                    df_linreg = df_recent[(df_recent['buy_signal'] == 1) | (df_recent['sell_signal'] == 1)]
                    if not df_linreg.empty:
                        for idx, row in df_linreg.iterrows():
                            linreg_results.append({
                                'Ticker': ticker,
                                'Date': idx,
                                'Buy Signal': row['buy_signal'],
                                'Sell Signal': row['sell_signal']
                            })
                    
                    # R² indicator cross signals filtering
                    df_r2 = df_recent[df_recent['cross_signal'] == 1]
                    if not df_r2.empty:
                        for idx, row in df_r2.iterrows():
                            r2_results.append({
                                'Ticker': ticker,
                                'Date': idx,
                                'Cross Signal': row['cross_signal']
                            })
            except Exception as e:
                logging.error(f"Failed to process data for ticker {ticker} on timeframe {tf}: {e}")
    
    # Save results to CSV files
    linreg_df = pd.DataFrame(linreg_results)
    linreg_df.to_csv('Regression_linreg_screener_results_2h.csv', index=False)
    logging.info("Saved Linear Regression screener results to CSV.")
    
    r2_df = pd.DataFrame(r2_results)
    r2_df.to_csv('Regression_cross_screener_results_2h.csv', index=False)
    logging.info("Saved R² screener results to CSV.")
    
    # ---------------------------
    # Send Telegram Alerts regardless of signals found
    # ---------------------------
    # Linear Regression Alert
    if not linreg_df.empty:
        logging.info("Sending Linear Regression signals alert.")
        message_linreg = (
            "2hr Screener Results:\n"
            "Buy = linreg(25) crosses above linreg(50)\n"
            "Sell = linreg(25) crosses below linreg(50)\n\n" +
            linreg_df.to_string(index=False)
        )
    else:
        logging.info("No Linear Regression signals found in the last 2 hours.")
        message_linreg = "2hr Screener Results:\nNo Linear Regression signals found in the last 2 hours."
    
    send_telegram_message(message_linreg)
    
    # Wait for 2 seconds between alerts
    time.sleep(2)
    
    # R² Indicator Alert
    if not r2_df.empty:
        logging.info("Sending R² signals alert.")
        message_r2 = (
            "2hr Screener Results:\n"
            "Signal: r2_smoothed (Length=25, AvgLen=3) crossing under 0.9\n\n" +
            r2_df.to_string(index=False)
        )
    else:
        logging.info("No R² signals found in the last 2 hours.")
        message_r2 = "2hr Screener Results:\nNo R² signals found in the last 2 hours."
    
    send_cross_telegram_message(message_r2)

############################
# 6. Entry Point
############################
if __name__ == "__main__":
    main()
