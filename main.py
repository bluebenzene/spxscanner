import os
import pytz
import requests
import pandas as pd
import numpy as np
import pandas_ta as ta
import yfinance as yf
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from dotenv import load_dotenv

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

# # Ensure the script runs only during US market hours.
if not is_us_market_open():
    print("The US market is currently closed. Script execution halted.")
else:
    load_dotenv()

    ############################
    # 2. Utility Functions
    ############################
    def get_sp500_tickers():
        """Fetches the S&P 500 tickers from Wikipedia."""
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'id': 'constituents'})
        tickers = [row.find('td').text.strip() for row in table.find_all('tr')[1:]]
        return tickers

    def send_telegram_message(message):
        """Sends a given message to a Telegram chat."""
        bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        data = {"chat_id": chat_id, "text": message}
        requests.post(url, data=data)

    ############################
    # 3. Setup: Fetch Tickers & Save to CSV
    ############################
    sp500_tickers = get_sp500_tickers()
    os.makedirs('stockdata', exist_ok=True)
    pd.DataFrame(sp500_tickers, columns=["Ticker"]).to_csv('stockdata/sp500_tickers.csv', index=False)
    tickers = pd.read_csv('stockdata/sp500_tickers.csv')['Ticker'].tolist()

    ############################
    # 4. Download Bulk Daily Data Using CachedLimiterSession & yf.Tickers
    ############################
    # Define timeframe: daily data over the past 90 days
    timeframes = {'1d': timedelta(days=90)}
    recent_period = 1  # Look back the last 1 day for signals
    end_date = datetime.now()

    # Set up bulk download with caching and rate limiting
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

    # Create a Tickers object with all tickers (space-separated) using the custom session.
    tickers_str = " ".join(tickers)
    dat = yf.Tickers(tickers_str, session=session)

    data = {}
    screener_results = []  # List to store screener signal results

    # Loop over the defined timeframe(s) (here only '1d')
    for tf, delta in timeframes.items():
        period_str = f"{delta.days}d"  # e.g., "90d"
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
                # Extract data for this ticker and make a copy
                df = his_data.xs(ticker, axis=1, level='Ticker').copy()
                # Flatten columns if they are a MultiIndex
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                # Ensure index is datetime and round OHLC values
                df.index = pd.to_datetime(df.index)
                for col in ['Open', 'High', 'Low', 'Close']:
                    if col in df.columns:
                        df.loc[:, col] = df[col].round(2)

                ############################
                # 5. Calculate Indicators & Signals
                ############################
                # Linear Regression Channels
                df['reg1'] = ta.linreg(df['Close'], length=10)
                df['reg2'] = ta.linreg(df['Close'], length=14)
                df['reg3'] = ta.linreg(df['Close'], length=30)

                # R-squared Calculation over a 14-day rolling window
                r2_length = 14
                df['r2_raw'] = df['Close'].rolling(window=r2_length).apply(
                    lambda x: np.corrcoef(x, np.arange(r2_length))[0, 1]**2 if len(x) == r2_length else np.nan,
                    raw=False
                )
                df['r2'] = df['r2_raw'] * 100
                df['r2_smoothed'] = df['r2'].rolling(window=3).mean()

                # Calculate RSI (14-day period) using pandas_ta; adds "RSI_14" column.
                df.ta.rsi(close='Close', length=14, append=True)

                # Define Buy and Sell Signals:
                # Buy when smoothed R² is high (> 90) and RSI is oversold (< 30)
                # Sell when smoothed R² is high (> 90) and RSI is overbought (> 70)
                df['buy_signal'] = np.where((df['r2_smoothed'] > 90) & (df['RSI_14'] < 30), 1, 0)
                df['sell_signal'] = np.where((df['r2_smoothed'] > 90) & (df['RSI_14'] > 70), 1, 0)

                # Save each ticker's data to CSV
                csv_filename = f'stockdata/{ticker}_{tf}_data.csv'
                df.to_csv(csv_filename)

                # -------------------------------
                # Filter for Recent Data (last 1 day) and Signal Conditions
                # -------------------------------
                recent_date_cutoff = df.index.max() - pd.Timedelta(days=recent_period)
                df_recent = df[df.index >= recent_date_cutoff]
                df_filtered = df_recent[(df_recent['buy_signal'] == 1) | (df_recent['sell_signal'] == 1)]
                if not df_filtered.empty:
                    for date, row in df_filtered.iterrows():
                        screener_results.append({
                            'Ticker': ticker,
                            'Date': date,
                            'Buy Signal': row['buy_signal'],
                            'Sell Signal': row['sell_signal']
                        })
            except Exception as e:
                print(f"Failed to process data for ticker {ticker} on timeframe {tf}: {e}")

    # Save screener results to CSV
    screener_df = pd.DataFrame(screener_results)
    screener_df.to_csv('screener_results_1d.csv', index=False)

    # Send results to Telegram if any signals found
    if not screener_df.empty:
        message = (
            "Daily Screener Results:\n"
            "Buy: R² > 90 and RSI_14 < 30\n"
            "Sell: R² > 90 and RSI_14 > 70\n\n" +
            screener_df.to_string(index=False)
        )
        send_telegram_message(message)

    # Display a sample of the screener results
    print("Daily Screener Results: R² > 90 and RSI_14")
    print(screener_df.tail())
