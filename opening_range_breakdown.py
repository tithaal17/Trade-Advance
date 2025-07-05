 
import sqlite3, config
import smtplib, ssl
import datetime
from datetime import timedelta,date
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus


context = ssl.create_default_context()

# Connect to database
connection = sqlite3.connect(config.DB_PATH)
connection.row_factory = sqlite3.Row
cursor = connection.cursor()

# Get Strategy ID
cursor.execute("""SELECT id FROM strategy WHERE name = "opening_range_breakdown" """)
strategy_id = cursor.fetchone()['id']

# Get Stocks linked to Strategy
cursor.execute("""
    SELECT symbol, name FROM stock
    JOIN stock_strategy ON stock_strategy.stock_id = stock.id
    WHERE stock_strategy.strategy_id = ?
""", (strategy_id,))
stocks = cursor.fetchall()

symbols = [stock['symbol'] for stock in stocks]

# Initialize Alpaca Trading Client
trading_client = TradingClient(config.API_KEY, config.SECRET_KEY, paper=True)

# Initialize Market Data Client
market_data = StockHistoricalDataClient(config.API_KEY, config.SECRET_KEY)

# Get Today's Date
today = date.today() if datetime.datetime.now(datetime.timezone.utc).hour >= 14 else date.today() - timedelta(days=1)

after_time = datetime.datetime(today.year, today.month, today.day, 9,45,0).isoformat() + "Z"

orders_request = GetOrdersRequest(
    status=QueryOrderStatus.ALL,
    limit=500,
    after=after_time
)
# Fetch existing orders
orders = trading_client.get_orders(filter=orders_request)
existing_order_symbols = [order.symbol for order in orders if order.status != 'canceled']

start_minute_bar = f"{today}T19:30:00"  # 9:30 AM EST = 7:00 PM IST
end_minute_bar = f"{today}T19:45:00"  # 9:45 AM EST = 7:15 PM IST


messages = []

# Fetch 1-minute bars for all stocks at once
request_params = StockBarsRequest(
    symbol_or_symbols=symbols,
    timeframe=TimeFrame.Minute,
    start=f"{today}T09:30:00Z",  # Ensure correct timestamp format
    end=f"{today}T16:00:00Z",  # Extend to full market hours if needed
    feed="iex" 
)
minute_bars = market_data.get_stock_bars(request_params).df

for symbol in symbols:
    try:
        print(f"Processing data for {symbol}...")

        if symbol not in minute_bars.index.get_level_values(0):
            print(f"No data found for {symbol}. Skipping.")
            continue

        stock_data = minute_bars.xs(symbol, level=0)

        opening_range_mask = (stock_data.index >= start_minute_bar) & (stock_data.index < end_minute_bar)
        opening_range_bars = stock_data.loc[opening_range_mask]

        opening_range_low = opening_range_bars['low'].min()
        opening_range_high = opening_range_bars['high'].max()
        opening_range = opening_range_high - opening_range_low
        
        after_opening_range_mask = stock_data.index >= end_minute_bar
        after_opening_range_bars = stock_data.loc[after_opening_range_mask]
       
        after_opening_range_breakdown = after_opening_range_bars[after_opening_range_bars['close'] < opening_range_low]
        

        if not after_opening_range_breakdown.empty:
            if symbol not in existing_order_symbols:
                limit_price = after_opening_range_breakdown.iloc[0]['close']

                message = f"selling short for {symbol} at {limit_price}, closed above {opening_range_high} \n\n{after_opening_range_breakdown.iloc[0]}\n\n"

                messages.append(message)

                print(message)

                # Submit bracket order
                order_request = LimitOrderRequest(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    qty=100,
                    limit_price=limit_price,
                    time_in_force=TimeInForce.DAY,
                    order_class=OrderClass.BRACKET,
                    take_profit=dict(limit_price=limit_price + opening_range),
                    stop_loss=dict(stop_price=limit_price - opening_range)
                )

                trading_client.submit_order(order_request)
            else:
                print(f"Already shorted for {symbol}, skipping.")

    except Exception as e:
        print(f"Error processing data for {symbol}: {e}")

#Send email notification
with smtplib.SMTP_SSL(config.EMAIL_HOST, config.EMAIL_PORT, context=context) as server:
    server.login(config.EMAIL_ADDRESS, config.EMAIL_PASSWORD)

    email_message = f"Subject: Trade Notifications for {today}\n\n"
    email_message += "\n\n".join(messages)

    server.sendmail(config.EMAIL_ADDRESS, config.EMAIL_ADDRESS, email_message)    
   