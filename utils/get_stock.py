#pip install yfinance -q
import pandas as pd
import yfinance as yf

df = yf.download('itub3.sa') # -> a ação buscada
df = df.drop(columns=['Adj Close', 'Volume'], axis = 1)
df = df.rename(columns={"Open": "Abertura", "High": "Máxima", 
      "Low": "Mínima", "Close": "Fechamento"})
df.to_csv('df_itub3.csv')