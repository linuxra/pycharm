import yfinance as yf
import pandas as pd

# get the list of all S&P500 stocks
table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
df = table[0]
stocks = ['MSFT','CRWD','TEAM']

for stock in stocks:
    data = yf.Ticker(stock)
    history = data.history(period='1mo') # Fetching for two days
    if history.empty or len(history) < 2:
        continue
    yesterday_close = history.iloc[0]['Close']
    today_close = history.iloc[-1]['Close']
    percent_change = ((today_close - yesterday_close) / yesterday_close) * 100
    if percent_change >= 1:
        print(f"{stock} has gained {percent_change}% today!")
