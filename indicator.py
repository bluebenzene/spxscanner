import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import pandas_ta as ta
import numpy as np
import requests
from bs4 import BeautifulSoup

# ---------------------------------------------
# OPTIONAL: Code to fetch S&P 500 tickers from Wikipedia
# Uncomment the following block if you want to update the ticker list dynamically.
# Otherwise, use your existing "sp500_tickers.csv" file.
#
# def get_sp500_tickers():
#     url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
#     response = requests.get(url)
#     soup = BeautifulSoup(response.text, 'html.parser')
#     table = soup.find('table', {'id': 'constituents'})
#     tickers = [row.find('td').text.strip() for row in table.find_all('tr')[1:]]
#     return tickers
#
# sp500_tickers = get_sp500_tickers()
# pd.DataFrame(sp500_tickers, columns=["Ticker"]).to_csv('sp500_tickers.csv', index=False)
# ---------------------------------------------

# Read the tickers from the CSV file
tickers_df = pd.read_csv('sp500_tickers.csv')
tickers = tickers_df['Ticker'].tolist()

# Define the timeframes and their corresponding date ranges.
# In this example, we're using only daily data for the past 180 days.
timeframes = {
    '1d': timedelta(days=180)
}

# Define the "recent period" for which we want to check signals (last 1 day)
recent_period = 1

# Function to download data for a given ticker and timeframe
def download_data(ticker, timeframe, start_date, end_date):
    # Download data using yfinance. The interval is set by timeframe.
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

        # Save each ticker's data to a CSV file for further inspection if needed
        df.to_csv(f'{ticker}_{tf}_data.csv')

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

# Save the overall screener results to a CSV file
screener_df = pd.DataFrame(screener_results)
screener_df.to_csv('screener_results.csv', index=False)

# Display a sample of the screener results
print("Screener results:")
print(screener_df.head())
