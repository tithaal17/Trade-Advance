


import sqlite3, config
from datetime import datetime, timedelta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import tulipy, numpy as np

# Register adapter to prevent Python 3.12 DeprecationWarning
sqlite3.register_adapter(datetime.date, lambda d: d.strftime("%Y-%m-%d"))

# Connect to SQLite database
connection = sqlite3.connect(config.DB_PATH)
connection.row_factory = sqlite3.Row
cursor = connection.cursor()

# Fetch all stocks from the database
cursor.execute("SELECT id, symbol FROM stock")
rows = cursor.fetchall()

# Prepare stock dictionaries
stock_dict = {row["symbol"]: row["id"] for row in rows}
symbols = list(stock_dict.keys())

# Initialize Alpaca Market Data Client
market_data = StockHistoricalDataClient(config.API_KEY, config.SECRET_KEY)

# Process stocks in batches
chunk_size = 200  
for i in range(0, len(symbols), chunk_size):
    symbol_chunk = symbols[i : i + chunk_size]

    try:
        new_rows = []

        for symbol in symbol_chunk:
            stock_id = stock_dict[symbol]

            # Get the last recorded date for this specific stock
            cursor.execute("SELECT MAX(date) FROM stock_price WHERE stock_id = ?", (stock_id,))
            last_recorded_date = cursor.fetchone()[0]

            if last_recorded_date:
                start_date = (datetime.strptime(last_recorded_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
            else:
                start_date = (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d")

            end_date = datetime.today().strftime("%Y-%m-%d")

            # Skip if already up-to-date
            if start_date >= end_date:
                continue

            # Fetch past 50 days of close prices from DB
            cursor.execute("""
                SELECT close FROM stock_price 
                WHERE stock_id = ? 
                ORDER BY date DESC LIMIT 50
            """, (stock_id,))
            past_closes = [row[0] for row in cursor.fetchall()]
            past_closes.reverse()  # Oldest first for indicator calculation

            # Fetch new data from Alpaca
            request_params = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start=start_date,
                end=end_date,
                feed="iex"
            )

            bars = market_data.get_stock_bars(request_params).df
            if bars.empty:
                continue

            # Prepare new close prices
            new_closes = [row["close"] for _, row in bars.iterrows()]
            
            # Merge past and new close prices
            full_closes = past_closes + new_closes
            full_closes_np = np.array(full_closes[-50:], dtype=np.float64)  # Keep only last 50 prices

            # Compute indicators
            sma_20 = tulipy.sma(full_closes_np, period=20)[-1] if len(full_closes_np) >= 20 else None
            sma_50 = tulipy.sma(full_closes_np, period=50)[-1] if len(full_closes_np) >= 50 else None
            rsi_14 = tulipy.rsi(full_closes_np, period=14)[-1] if len(full_closes_np) >= 14 else None

            # Insert new data into stock_price table
            for index, row in bars.iterrows():
                date = index[1].date()
                new_rows.append((stock_id, date, row["open"], row["high"], row["low"], row["close"], row["volume"], sma_20, sma_50, rsi_14))

        # Insert all new rows into the database
        if new_rows:
            cursor.executemany("""
                INSERT INTO stock_price (stock_id, date, open, high, low, close, volume, sma_20, sma_50, rsi_14) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(stock_id, date) DO NOTHING;
            """, new_rows)

        print(f"✅ Successfully processed batch {i//chunk_size + 1}")

    except Exception as batch_error:
        print(f"⚠️ Error processing batch {i//chunk_size + 1}: {batch_error}")

# Commit changes and close DB connection
connection.commit()
cursor.close()
connection.close()

print("✅ Stock price database successfully updated!")




# import sqlite3, config
# from datetime import datetime, timedelta
# from alpaca.data.historical import StockHistoricalDataClient
# from alpaca.data.requests import StockBarsRequest
# from alpaca.data.timeframe import TimeFrame
# import tulipy, numpy as np

# # Register adapter to prevent Python 3.12 DeprecationWarning
# sqlite3.register_adapter(datetime.date, lambda d: d.strftime("%Y-%m-%d"))

# # Connect to SQLite database
# connection = sqlite3.connect(config.DB_PATH)
# connection.row_factory = sqlite3.Row
# cursor = connection.cursor()

# # Fetch all stocks from the database
# cursor.execute("SELECT id, symbol FROM stock")
# rows = cursor.fetchall()

# # Prepare stock dictionaries
# symbols = []
# stock_dict = {}

# for row in rows:
#     symbol = row["symbol"]
#     symbols.append(symbol)
#     stock_dict[symbol] = row["id"]

# # Fetch existing stock_id and date pairs to prevent duplicates
# cursor.execute("SELECT stock_id, date FROM stock_price")
# existing_data_set = set((row["stock_id"], row["date"]) for row in cursor.fetchall())

# # Initialize Alpaca Market Data Client
# market_data = StockHistoricalDataClient(config.API_KEY, config.SECRET_KEY)

# # Find last recorded date in stock_price table
# cursor.execute("SELECT MAX(date) FROM stock_price")
# last_recorded_date = cursor.fetchone()[0]

# if last_recorded_date:
#     start_date = (datetime.strptime(last_recorded_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
# else:
#     start_date = (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d")

# end_date = datetime.today().strftime("%Y-%m-%d")

# # Skip if already up-to-date
# if start_date >= end_date:
#     print("No new data to update.")
# else:
#     print(f"Fetching data from {start_date} to {end_date} for {len(symbols)} stocks")

#     chunk_size = 200  # ✅ Process in batches
#     for i in range(0, len(symbols), chunk_size):
#         symbol_chunk = symbols[i : i + chunk_size]
        
#         try:
#             # Fetch stock bars in batch
#             request_params = StockBarsRequest(
#                 symbol_or_symbols=symbol_chunk,
#                 timeframe=TimeFrame.Day,
#                 start=start_date,
#                 end=end_date,
#                 feed="iex"  # ✅ Use IEX data
#             )

#             bars = market_data.get_stock_bars(request_params).df

#             if bars.empty:
#                 continue

#             new_rows = []
#             close_prices = {}

#             # Prepare close prices per symbol for indicators
#             for index, row in bars.iterrows():
#                 symbol = index[0]
#                 date = index[1].date()  # Extract date from multi-index
                
#                 if symbol in stock_dict:  
#                     stock_id = stock_dict[symbol]

#                     # ✅ Store close prices for indicators calculation
#                     if symbol not in close_prices:
#                         close_prices[symbol] = []
#                     close_prices[symbol].append(row["close"])

#                     # ✅ Check if the record already exists before inserting
#                     if (stock_id, date) not in existing_data_set:
#                         new_rows.append((stock_id, date, row["open"], row["high"], row["low"], row["close"], row["volume"], None, None, None))
#                         existing_data_set.add((stock_id, date))

#             # Calculate indicators & update rows
#             final_rows = []
#             for symbol, closes in close_prices.items():
#                 stock_id = stock_dict[symbol]

#                 closes_np = np.array(closes, dtype=np.float64)

#                 sma_20 = tulipy.sma(closes_np, period=20)[-1] if len(closes) >= 20 else None
#                 sma_50 = tulipy.sma(closes_np, period=50)[-1] if len(closes) >= 50 else None
#                 rsi_14 = tulipy.rsi(closes_np, period=14)[-1] if len(closes) >= 14 else None

#                 for row in new_rows:
#                     if row[0] == stock_id:
#                         final_rows.append((*row[:-3], sma_20, sma_50, rsi_14))  # Replace last 3 None values

#             if final_rows:
#                 cursor.executemany("""
#                     INSERT INTO stock_price (stock_id, date, open, high, low, close, volume, sma_20, sma_50, rsi_14) 
#                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
#                 """, final_rows)

#             print(f"✅ Successfully processed batch {i//chunk_size + 1}.")

#         except Exception as e:
#             print(f"⚠️ Error fetching batch {i//chunk_size + 1}: {e}")

# # Commit changes and close DB connection
# connection.commit()
# cursor.close()
# connection.close()

# print("✅ Stock price database successfully updated!")

