import pandas as pd
import requests
from bs4 import BeautifulSoup

def get_russell_2000_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_Russell_2000_companies'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find the table containing the tickers
    table = soup.find('table', {'class': 'wikitable sortable'})
    
    if table is None:
        raise ValueError("Could not find the table containing the tickers.")
    
    # Extract tickers from the table
    rows = table.find_all('tr')[1:]  # Skip the header row
    tickers = []
    for row in rows:
        ticker = row.find_all('td')[0].text.strip()
        tickers.append(ticker)
    return tickers

# Get the Russell 2000 tickers
try:
    russell_2000_tickers = get_russell_2000_tickers()
    # Display the tickers
    print("Russell 2000 (Small Cap) Tickers:")
    print(russell_2000_tickers)

    # Save the tickers to a CSV file
    pd.DataFrame(russell_2000_tickers, columns=["Ticker"]).to_csv('russell_2000_tickers.csv', index=False)
except Exception as e:
    print(f"An error occurred: {e}")
