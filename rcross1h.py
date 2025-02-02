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
    from datetime import datetime, timedelta
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
        '1h': timedelta(days=30),
        # '1d': timedelta(days=90),
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

    # Applying the custom indicator and generating buy/sell signals
    for tf in timeframes:
        for ticker in data[tf]:
            df = data[tf][ticker]
            
                    # If the DataFrame columns are a MultiIndex, flatten them
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            
            # Round OHLC values to two decimal points
            df['Open'] = df['Open'].round(2)
            df['High'] = df['High'].round(2)
            df['Low'] = df['Low'].round(2)
            df['Close'] = df['Close'].round(2)
            
        # --- Existing calculations ---
            # Linear Regression Curves
            
            df['reg1'] = ta.linreg(df['Close'], length=25)
            df['reg2'] = ta.linreg(df['Close'], length=50)
            

            df['buy_signal'] = np.where(
                (df['reg1'] > df['reg2']) & (df['reg1'].shift(1) <= df['reg2'].shift(1)),
                1,
                0
            )
            
            # Sell Signal: When the 25-period line (reg1) crosses below the 50-period line (reg2)
            df['sell_signal'] = np.where(
                (df['reg1'] < df['reg2']) & (df['reg1'].shift(1) >= df['reg2'].shift(1)),
                1,
                0
            )
                    
            # Save each ticker's data to a CSV file in the stockdata folder
            df.to_csv(f'stockdata/{ticker}_{tf}_data.csv')

            
            hoursback =1 
            # Filter for recent periods
            recent_date_cutoff = df.index.max() - pd.Timedelta(hours=hoursback)
            df = df[df.index >= recent_date_cutoff]
            
            # Filter rows with buy or sell signals
            df_filtered = df[(df['buy_signal'] == 1) | (df['sell_signal'] == 1)]
            
            # Append results to screener list
            if not df_filtered.empty:
                for index, row in df_filtered.iterrows():
                    screener_results.append({
                        'Ticker': ticker,
                        'Date': index,
                        'Buy Signal': row['buy_signal'],
                        'Sell Signal': row['sell_signal']
                    })

    # Save screener results to a CSV file
    screener_df = pd.DataFrame(screener_results)
    screener_df.to_csv('Regression_cross_screener_results_1h.csv', index=False)


    # Send results to Telegram
    if not screener_df.empty:
        message = "linear regression crossover occurs 25,50\n 1hr timeframe \n" + screener_df.to_string(index=False)
        send_telegram_message(message)

    # Display a sample of the screener results
    print("RESULT:")
    print(screener_df.tail())
