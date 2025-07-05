# database schema:

import sqlite3

connection = sqlite3.connect('trading.db')

cursor = connection.cursor()

cursor.execute("""
               CREATE TABLE IF NOT EXISTS stock(
                    id INTEGER PRIMARY KEY,
                    symbol TEXT NOT NULL UNIQUE,
                    name TEXT NOT NULL,
                    exchange TEXT NOT NULL
                )
            """)

cursor.execute("""
               CREATE TABLE IF NOT EXISTS stock_price(
                    id INTEGER PRIMARY KEY,
                    stock_id INTEGER,
                    date NOT NULL,
                    open NOT NULL,
                    high NOT NULL,
                    low NOT NULL,
                    close NOT NULL,
                    volume NOT NULL,
                    FOREIGN KEY(stock_id) REFERENCES stock(id)
                )
            """)

connection.commit()

# populate_stock.py


import sqlite3, config
import alpaca_trade_api as tradeapi

connection = sqlite3.connect(config.DB_PATH)
connection.row_factory = sqlite3.Row

cursor = connection.cursor()

cursor.execute("""
                SELECT symbol, name FROM stock
               """)

rows = cursor.fetchall()
symbols = [row['symbol'] for row in rows]

api = tradeapi.REST(config.API_KEY, config.SECRET_KEY, base_url=config.API_URL)

assets = api.list_assets()

for asset in assets:
    try:
        if asset.status == 'active' and asset.tradable and asset.symbol not in symbols and "/" not in asset.symbol:
            
            print(f"Added a new stock {asset.symbol} {asset.name}")
            
            cursor.execute("INSERT INTO stock (symbol, name, exchange) VALUES (?, ?, ?)", (asset.symbol, asset.name, asset.exchange))

    except Exception as e:
        print(asset.symbol)
        print(e)

connection.commit()
print("success")

# populate_prices.py

import sqlite3, config
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta

connection = sqlite3.connect(config.DB_PATH)
connection.row_factory = sqlite3.Row

cursor = connection.cursor()

cursor.execute("""
                SELECT id,symbol, name FROM stock
               """)

rows = cursor.fetchall()

symbols = []
stock_dict = {}
for row in rows:
    symbol = row['symbol']
    symbols.append(symbol)
    stock_dict[symbol] = row['id']

api = tradeapi.REST(config.API_KEY, config.SECRET_KEY, base_url=config.API_URL)

start_date = (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d") 
end_date = datetime.today().strftime("%Y-%m-%d")

chunk_size = 200
for i in range(0, len(symbols), chunk_size): 
    symbol_chunk = symbols[i:i+chunk_size]
    barsets = api.get_bars(symbol_chunk, timeframe="1day", start=start_date, end=end_date)

    for bar in barsets:  # ✅ Iterate directly over the list
        print(f"Processing symbol {bar.S}")  
        cursor.execute("""
            INSERT INTO stock_price(stock_id, date, open, high, low, close, volume) 
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (stock_dict[bar.S], bar.t.date(), bar.o, bar.h, bar.l, bar.c, bar.v))


connection.commit()
cursor.close() 
connection.close()

# populate stock using yf

import sqlite3
import yfinance as yf
import config
import pandas as pd


# List of Indices and ETFs to track
index_etfs = [
    "^IXIC",  # Nasdaq Composite
    "^DJI",   # Dow Jones Industrial Average
    "^NSEI",  # NIFTY 50
    "^BSESN", # SENSEX
    "SPY",    # S&P 500 ETF
    "QQQ",    # Nasdaq 100 ETF
    "IVV",    # S&P 500 Index Fund
    "VOO"     # Vanguard S&P 500 ETF
]

# Fetch stock lists for S&P 500, Nasdaq 100, and NIFTY 100

# ✅ Load tickers from CSV files
sp500_df = pd.read_csv("SP500.csv")
nasdaq100_df = pd.read_csv("NASDAQ100.csv")  # Add your NASDAQ 100 CSV

sp500_tickers = set(sp500_df["Symbol"].str.strip().tolist())
nasdaq100_tickers = set(nasdaq100_df["Symbol"].str.strip().tolist())

# ✅ Remove duplicates between S&P 500 and NASDAQ 100
nasdaq100_tickers -= sp500_tickers

def get_nifty100_tickers():
    url = "https://archives.nseindia.com/content/indices/ind_nifty100list.csv"
    df = pd.read_csv(url)
    return list(df["Symbol"].str.replace("&", "-"))  # NSE uses "&" which is problematic

nifty100_tickers = set(get_nifty100_tickers())

# Remove duplicates (stocks appearing in multiple indices)
all_tickers = list(sp500_tickers | nasdaq100_tickers | nifty100_tickers)

# Connect to the database
connection = sqlite3.connect(config.DB_PATH)
connection.row_factory = sqlite3.Row
cursor = connection.cursor()


# Fetch existing symbols from database
cursor.execute("SELECT symbol FROM stock")
existing_symbols = {row[0] for row in cursor.fetchall()}

# Insert new stocks into the database
for symbol in all_tickers + index_etfs:
    if symbol not in existing_symbols:
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            name = info.get("shortName", symbol)
            exchange = info.get("exchange", "Unknown")
            sector = info.get("sector", "Unknown")

            print(f"Adding {symbol} - {name} ({exchange})")
            cursor.execute("INSERT INTO stock (symbol, name, exchange, sector) VALUES (?, ?, ?, ?)", (symbol, name, exchange, sector))
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")

connection.commit()
cursor.close()
connection.close()

# populate price with yf

import sqlite3
import yfinance as yf
import config
from datetime import datetime, timedelta

# Connect to the database
connection = sqlite3.connect(config.DB_PATH)
connection.row_factory = sqlite3.Row
cursor = connection.cursor()

# Fetch all stocks from database
cursor.execute("SELECT id, symbol FROM stock")
stocks = cursor.fetchall()

# Fetch the last stored price date for each stock

cursor.execute("""
    SELECT stock_id, MAX(date) FROM stock_price GROUP BY stock_id
""")
last_dates = {row[0]: row[1] for row in cursor.fetchall()}

# ✅ Fetch historical data (5 years)
for stock_id, symbol in stocks:
    last_date = last_dates.get(stock_id, None)

    if last_date:
        start_date = (datetime.strptime(last_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        start_date = (datetime.today() - timedelta(days=5*365)).strftime("%Y-%m-%d")  # Full 5 years

    print(f"Fetching 5-year historical prices for {symbol} from {start_date}...")

    try:
        data = yf.download(symbol, start=start_date, progress=False)

        if data.empty:
            print(f"⚠️ No data found for {symbol}, skipping...")
            continue

        # ✅ Convert data to tuples for SQLite
        records = [
            (
                stock_id, 
                date.strftime("%Y-%m-%d"),  # Convert to string format
                float(row["Open"]),   # Convert to Python float
                float(row["High"]),   # Convert to Python float
                float(row["Low"]),    # Convert to Python float
                float(row["Close"]),  # Convert to Python float
                int(row["Volume"])    # Convert to Python int
            )
            for date, row in data.iterrows()
        ]

        # ✅ Insert data into the database
        cursor.executemany(
            "INSERT INTO stock_price (stock_id, date, open, high, low, close, volume) VALUES (?, ?, ?, ?, ?, ?, ?)",
            records
        )

    except Exception as e:
        print(f"❌ Error fetching price data for {symbol}: {e}")

# ✅ Commit changes
connection.commit()
cursor.close()
connection.close()
print("✅ Stock prices updated successfully.")