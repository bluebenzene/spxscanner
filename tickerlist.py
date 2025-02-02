import pandas as pd
import requests
from bs4 import BeautifulSoup

# Function to fetch S&P 500 tickers from Wikipedia
def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'id': 'constituents'})
    tickers = [row.find('td').text.strip() for row in table.find_all('tr')[1:]]
    return tickers[:10]# Get only the first 10 tickers

# Get the S&P 500 tickers
sp500_tickers = get_sp500_tickers()

# Display the tickers
print("S&P 500 Tickers:")
print(sp500_tickers)

# Save the tickers to a CSV file
pd.DataFrame(sp500_tickers, columns=["Ticker"]).to_csv('sp500_tickers.csv', index=False)
