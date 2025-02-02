
import pytz
from datetime import datetime, timedelta

def is_us_market_open():
    """Checks if the current time is within US market hours (9:30 AM - 4:00 PM ET) and not on weekends."""
    eastern = pytz.timezone('US/Eastern')
    now = datetime.now(eastern)
    
    # Market closed on weekends
    if now.weekday() in [5, 6]:  # Saturday (5) and Sunday (6)
        return False
    
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    
    return market_open <= now <= market_close


# Ensure the script runs only during US market hours
if not is_us_market_open():
    print("The US market is currently closed. Script execution halted.")
else:
    import pandas as pd
    import yfinance as yf
    import pandas_ta as ta
    import numpy as np
    import requests
    from bs4 import BeautifulSoup
    import os
    from dotenv import load_dotenv
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

    # Ensure the stockdata directory exists
    os.makedirs('stockdata', exist_ok=True)

    # Save the tickers to a CSV file in the stockdata folder
    pd.DataFrame(sp500_tickers, columns=["Ticker"]).to_csv('stockdata/sp500_tickers.csv', index=False)

    # Read the tickers from the CSV file in the stockdata folder
    tickers_df = pd.read_csv('stockdata/sp500_tickers.csv')
    tickers = tickers_df['Ticker'].tolist()

    # Define the timeframes and their corresponding date ranges
    timeframes = {
        # '15m': timedelta(days=60),
        # '1h': timedelta(days=730),
        '1d': timedelta(days=90),
        # '1wk': timedelta(days=3*365)
    }

    # Define the recent period (in days)
    recent_period = 1

    # Function to download data for a given ticker and timeframe
    def download_data(ticker, timeframe, start_date, end_date):
        return yf.download(ticker, start=start_date, end=end_date, interval=timeframe)

    # Dictionary to store data for each timeframe
    data = {tf: {} for tf in timeframes}

    # Download historical data for each ticker and timeframe
    end_date = datetime.now()
    for ticker in tickers:
        for tf, delta in timeframes.items():
            start_date = end_date - delta
            try:
                df = download_data(ticker, tf, start_date, end_date)
                if df.empty:
                    print(f"No data available for {ticker} on timeframe {tf}. Skipping.")
                    continue
                data[tf][ticker] = df
            except Exception as e:
                print(f"Failed to download data for {ticker} with timeframe {tf}: {e}")


    # List to store screener results
    screener_results = []

    # Process each ticker's data for each timeframe
    for tf in timeframes:
        for ticker, df in data[tf].items():
            # Ensure the index is in datetime format (if not already)
            df.index = pd.to_datetime(df.index)
            
            # Round OHLC values to two decimal points
            for col in ['Open', 'High', 'Low', 'Close']:
                if col in df.columns:
                    df[col] = df[col].round(2)

            # -------------------------------
            # Calculate Linear Regression Channels
            # -------------------------------
            df['reg1'] = ta.linreg(df['Close'], length=10)
            df['reg2'] = ta.linreg(df['Close'], length=14)
            df['reg3'] = ta.linreg(df['Close'], length=30)

            # -------------------------------
            # R-squared Calculation
            # -------------------------------
            r2_length = 14
            # Calculate the R-squared value for a rolling window
            df['r2_raw'] = df['Close'].rolling(window=r2_length).apply(
                lambda x: np.corrcoef(x, np.arange(r2_length))[0, 1]**2 if len(x) == r2_length else np.nan,
                raw=False
            )
            # Normalize to a scale of 0 to 100 and smooth over 3 periods
            df['r2'] = df['r2_raw'] * 100  
            df['r2_smoothed'] = df['r2'].rolling(window=3).mean()

            # -------------------------------
            # Flatten the columns if they are a MultiIndex
            # -------------------------------
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # -------------------------------
            # Calculate RSI using pandas_ta (14-day period)
            # -------------------------------
            df.ta.rsi(close='Close', length=14, append=True)  # This adds an "RSI_14" column

            # -------------------------------
            # Define Buy and Sell Signals
            # -------------------------------
            # A buy signal is generated if the smoothed R-squared is high ( > 90 )
            # and the RSI is oversold ( < 30 )
            df['buy_signal'] = np.where((df['r2_smoothed'] > 90) & (df['RSI_14'] < 30), 1, 0)
            # A sell signal is generated if the smoothed R-squared is high ( > 90 )
            # and the RSI is overbought ( > 70 )
            df['sell_signal'] = np.where((df['r2_smoothed'] > 90) & (df['RSI_14'] > 70), 1, 0)

            # Save each ticker's data to a CSV file in the stockdata folder
            df.to_csv(f'stockdata/{ticker}_{tf}_data.csv')

            # -------------------------------
            # Filter for Recent Data and Signal Conditions
            # -------------------------------
            recent_date_cutoff = df.index.max() - pd.Timedelta(days=recent_period)
            df_recent = df[df.index >= recent_date_cutoff]
            df_filtered = df_recent[(df_recent['buy_signal'] == 1) | (df_recent['sell_signal'] == 1)]
            
            # If any signal was found in the recent period, add it to the results
            if not df_filtered.empty:
                for date, row in df_filtered.iterrows():
                    screener_results.append({
                        'Ticker': ticker,
                        'Date': date,
                        'Buy Signal': row['buy_signal'],
                        'Sell Signal': row['sell_signal']
                    })

    # Save screener results to a CSV file
    screener_df = pd.DataFrame(screener_results)
    screener_df.to_csv('screener_results_1d.csv', index=False)

    # Send results to Telegram
    if not screener_df.empty:
        message = "Daily screener Results:\n Buy=R2>90 and rsi14<30 \n Sell=R2>90 and rsi14>70 \n" + screener_df.to_string(index=False)
        send_telegram_message(message)

    # Display a sample of the screener results
    # print("R-squared is high ( > 90 ) and RSI is oversold ( < 30 ):")
    # print(screener_df.head())
