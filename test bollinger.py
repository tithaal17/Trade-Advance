import sqlite3, config
import alpaca_trade_api as tradeapi
import datetime,tulipy
import smtplib, ssl
from helpers import recommend_quantity

context = ssl.create_default_context()
# Connect to database
connection = sqlite3.connect(config.DB_PATH)
connection.row_factory = sqlite3.Row
cursor = connection.cursor()

# Get Strategy ID
cursor.execute("""SELECT id FROM strategy WHERE name = "bollinger_bands" """)
strategy_id = cursor.fetchone()['id']

# Get Stocks linked to Strategy
cursor.execute("""
    SELECT symbol, name FROM stock
    JOIN stock_strategy ON stock_strategy.stock_id = stock.id
    WHERE stock_strategy.strategy_id = ?
""", (strategy_id,))
stocks = cursor.fetchall()

symbols = [stock['symbol'] for stock in stocks]

# Initialize Alpaca API
api = tradeapi.REST(config.API_KEY, config.SECRET_KEY, base_url=config.API_URL)

# Get Today's Date
today = datetime.date.today().strftime("%Y-%m-%d")

orders = api.list_orders(status='all',limit=500, after=today)

existing_order_symbols = [order.symbol for order in orders if order.status != 'canceled']

start_minute_bar = f"{today} 9:30:00"
end_minute_bar = f"{today} 16:00:00"

messages = []

for symbol in symbols:
    
    # Fetch 1-minute bars with IEX data feed
    minute_bars = api.get_bars(symbol, '1Min', start=today, end=today, feed='iex').df

    market_open_mask = (minute_bars.index >= start_minute_bar) & (minute_bars.index < end_minute_bar)
    market_open_bars = minute_bars.loc[market_open_mask] 

    if len(market_open_bars) >= 20: 
        closes = market_open_bars.close.values

        lower, middle, upper = tulipy.bbands(closes, 20, 2)

        current_candle = market_open_bars.iloc[-1]
        previous_candle =  market_open_bars.iloc[-2]

        if current_candle.close > lower[-1] and previous_candle.close < lower[-2]:
            print(f"{symbol} closed above lower bollinger band")
            print(current_candle)

            if symbol not in  existing_order_symbols:

                limit_price = current_candle.close
                candle_range = current_candle.high - current_candle.low

                messages.append(f"placing order for {symbol} at {limit_price}, closed_above {lower[-2]} \n\n")

                print(f"placing order for {symbol} at {limit_price}")

                api.submit_order(
                    symbol=symbol,
                    side='buy',
                    type='limit',
                    qty=recommend_quantity(limit_price),
                    time_in_force='day',
                    order_class='bracket',
                    limit_price= limit_price,
                    take_profit=dict(
                        limit_price = limit_price + (candle_range * 3),
                    ),
                    stop_loss=dict(
                        stop_price = previous_candle.low,
                        
                    )
                )
            else:
                print(f"Already ordered for {symbol}, skipping")

