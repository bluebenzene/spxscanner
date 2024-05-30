import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup

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

# Save the tickers to a CSV file
pd.DataFrame(sp500_tickers, columns=["Ticker"]).to_csv('sp500_tickers.csv', index=False)

# Read the tickers from the CSV file
tickers_df = pd.read_csv('sp500_tickers.csv')
tickers = tickers_df['Ticker'].tolist()

# Calculate the date range for the past 3 years
end_date = datetime.now()
start_date = end_date - timedelta(days=3*365)

# Download historical OHLC data for each ticker
data = yf.download(tickers, start=start_date, end=end_date, interval='1d', group_by='ticker')

# Save the data to a CSV file
data.to_csv('sp500_ohlc_data.csv')

# Display a sample of the data
print(data.head())
