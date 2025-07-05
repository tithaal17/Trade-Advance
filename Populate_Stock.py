
import sqlite3, config
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.client import TradingClient

# Register a date adapter to prevent Python 3.12 warnings
import datetime
sqlite3.register_adapter(datetime.date, lambda d: d.strftime("%Y-%m-%d"))

# Connect to SQLite database
connection = sqlite3.connect(config.DB_PATH)
connection.row_factory = sqlite3.Row
cursor = connection.cursor()

# Fetch existing stock symbols
cursor.execute("SELECT symbol FROM stock")
rows = cursor.fetchall()
symbols = {row['symbol'] for row in rows}  # Use a set for faster lookups

# Initialize Alpaca Trading Client
trading_client = TradingClient(config.API_KEY, config.SECRET_KEY, paper=True)

# Fetch all assets
assets = trading_client.get_all_assets()

# Iterate and add new assets
for asset in assets:
    try:
        if asset.status == 'active' and asset.tradable and asset.shortable and asset.symbol not in symbols and "/" not in asset.symbol:
            print(f"Adding new stock: {asset.symbol} - {asset.name}")

            cursor.execute("INSERT INTO stock (symbol, name, exchange) VALUES (?, ?, ?)", 
                           (asset.symbol, asset.name, asset.exchange))

    except Exception as e:
        print(f"Error with {asset.symbol}: {e}")

# Commit changes and close DB connection
connection.commit()
cursor.close()
connection.close()

print("✅ Successfully updated stock database.")
