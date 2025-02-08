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
    
    import pytz
    from datetime import datetime, timedelta
    import pandas as pd
    import yfinance as yf
    import pandas_ta as ta
    import numpy as np
    import requests
    from bs4 import BeautifulSoup
    import os

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

    # Ensure the stockdata directory exists
    os.makedirs('stockdata', exist_ok=True)

    # Save the tickers to a CSV file in the stockdata folder
    pd.DataFrame(sp500_tickers, columns=["Ticker"]).to_csv('stockdata/sp500_tickers.csv', index=False)

    # Read the tickers from the CSV file in the stockdata folder
    tickers_df = pd.read_csv('stockdata/sp500_tickers.csv')
    tickers = tickers_df['Ticker'].tolist()

    # Define the timeframes and their corresponding date ranges
    timeframes = {
        '1h': timedelta(days=30),
    }

    # Define the recent period (in hours)
    recent_period = 2

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
                
                # Flatten MultiIndex columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                # Check for required columns
                required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                if not all(col in df.columns for col in required_columns):
                    print(f"Missing columns in {ticker} data. Skipping.")
                    continue
                
                # Resample 1-hour data to 2-hour data
                df_resampled = df.resample('2h').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }).dropna()
                
                data[tf][ticker] = df_resampled
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
            df['reg1'] = ta.linreg(df['Close'], length=25)
            df['reg2'] = ta.linreg(df['Close'], length=50)
            
            # Buy/Sell Signals
            df['buy_signal'] = np.where(
                (df['reg1'] > df['reg2']) & (df['reg1'].shift(1) <= df['reg2'].shift(1)),
                1,
                0
            )
            df['sell_signal'] = np.where(
                (df['reg1'] < df['reg2']) & (df['reg1'].shift(1) >= df['reg2'].shift(1)),
                1,
                0
            )
            
            # Save each ticker's data to a CSV file in the stockdata folder
            df.to_csv(f'stockdata/{ticker}_{tf}_data.csv')

            # Filter for recent periods
            recent_date_cutoff = df.index.max() - pd.Timedelta(hours=recent_period)
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
    screener_df.to_csv('Regression_cross_screener_results_2h.csv', index=False)


    # Send results to Telegram
    if not screener_df.empty:
        message = "Hourly screener Results:\n Buy=linreg 25 crossover linreg 50 \n Sell=linreg 25 cross below linreg 50 \n" + screener_df.to_string(index=False)
        send_telegram_message(message)

    # Display a sample of the screener results
    print("Hourly screener Results:\n Buy=linreg 25 crossover linreg 50 \n Sell=linreg 25 cross below linreg 50 \n")
    print(screener_df.tail())
