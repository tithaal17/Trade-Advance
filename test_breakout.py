import sqlite3, config
import alpaca_trade_api as tradeapi
import datetime
import smtplib, ssl

context = ssl.create_default_context()
# Connect to database
connection = sqlite3.connect(config.DB_PATH)
connection.row_factory = sqlite3.Row
cursor = connection.cursor()

# Get Strategy ID
cursor.execute("""SELECT id FROM strategy WHERE name = "opening_range_breakout" """)
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

orders = api.list_orders(status='all',limit=500, after=f"{today}T13:30:002")

existing_order_symbols = [order.symbol for order in orders]

start_minute_bar = f"{today} 9:30:00-4:00"
end_minute_bar = f"{today} 9:45:00-4:00"

messages = []

for symbol in symbols:
    
    # Fetch 1-minute bars with IEX data feed
    minute_bars = api.get_bars(symbol, '1Min', start=today, end=today, feed='iex').df
    
    opening_range_mask = (minute_bars.index >= start_minute_bar) & (minute_bars.index < end_minute_bar)
    opening_range_bars = minute_bars.loc(opening_range_mask)
 
    opening_range_low = opening_range_bars['low'].min()
    opening_range_high = opening_range_bars['high'].max()
    opening_range = opening_range_high - opening_range_low

    after_opening_range_mask = minute_bars.index >= end_minute_bar
    after_opening_range_bars = minute_bars.loc[after_opening_range_mask]

    after_opening_range_breakout = after_opening_range_bars[after_opening_range_bars['close'] >opening_range_high ]

    if not after_opening_range_breakout.empty:
        if symbol not in  existing_order_symbols:

            limit_price = after_opening_range_breakout.iloc[0]['close']
            
            messages.append(f"placing order for {symbol} at {limit_price}, closed_above {opening_range_high} \n\n{after_opening_range_breakout.iloc[0]}\n\n")

            print(f"placing order for {symbol} at {limit_price}, closed_above {opening_range_high}\n {after_opening_range_breakout.iloc[0]}")

            api.submit_order(
                symbol=symbol,
                side='buy',
                type='limit',
                qty='100',
                time_in_force='day',
                order_class='bracket',
                limit_price= limit_price,
                take_profit=dict(
                    limit_price = limit_price + opening_range,
                ),
                stop_loss=dict(
                    stop_price = limit_price - opening_range,
                    
                )
            )
        else:
            print(f"Already ordered for {symbol}, skipping")

with smtplib.SMTP_SSL(config.EMAIL_HOST, config.EMAIL_PORT, context=context) as server:

    server.login(config.EMAIL_ADDRESS, config.EMAIL_PASSWORD)

    email_message = f"Subject : Trade Notifications for {today}\n\n"
    email_message += "\n\n.join messages"

    server.sendmail(config.EMAIL_ADDRESS, config.EMAIL_ADDRESS, email_message)    
    server.sendmail(config.EMAIL_ADDRESS, config.EMAIL_SMS, email_message)  