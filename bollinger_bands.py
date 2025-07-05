import sqlite3
import config
import datetime
import tulipy as ti
import smtplib
import ssl
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus
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

# Initialize Alpaca Clients
trading_client = TradingClient(config.API_KEY, config.SECRET_KEY, paper=True)
data_client = StockHistoricalDataClient(config.API_KEY, config.SECRET_KEY)

# Get Today's Date
current_date = "2025-02-25"
today = datetime.datetime.strptime(current_date, "%Y-%m-%d").strftime("%Y-%m-%d")

orders_request = GetOrdersRequest(
    status=QueryOrderStatus.ALL,
    limit=500,
    after=today
)

orders = trading_client.get_orders(filter=orders_request)
existing_order_symbols = [order.symbol for order in orders if order.status != 'canceled']

# Define market hours
start_minute_bar = f"{today}T09:30:00"
end_minute_bar = f"{today}T16:00:00"

messages = []

for symbol in symbols:
    try:
        # Fetch 1-minute bars
        request_params = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute,
            start=f"{today}T09:30:00Z",  # Ensure correct timestamp format
            end=f"{today}T16:00:00Z",
            feed="iex" 
        )
        bars = data_client.get_stock_bars(request_params).df

        if symbol in bars.index:
            bars = bars.loc[symbol]  # Extract data for symbol

            market_open_mask = (bars.index >= start_minute_bar) & (bars.index < end_minute_bar)
            market_open_bars = bars.loc[market_open_mask]

            if len(market_open_bars) >= 20:
                closes = market_open_bars.close.values

                lower, middle, upper = ti.bbands(closes, 20, 2)
                atr_values = ti.atr(
                    market_open_bars.high.values,
                    market_open_bars.low.values,
                    closes,
                    period=14
                )
                print(lower)
                print(middle)
                print(upper)
                current_candle = market_open_bars.iloc[-1]
                previous_candle = market_open_bars.iloc[-2]
                current_atr = atr_values[-1]
                print(current_atr)
                # Long Entry Condition
                if current_candle.close > lower[-1] and previous_candle.close < lower[-2]:
                    print(f"{symbol} closed above lower Bollinger Band")

                    if symbol not in existing_order_symbols:
                        limit_price = current_candle.close
                        candle_range = current_candle.high - current_candle.low

                        messages.append(f"Placing long order for {symbol} at {limit_price} \n\n")

                        print(f"Placing long order for {symbol} at {limit_price}")

                        trading_client.submit_order(
                            symbol=symbol,
                            side='buy',
                            type='limit',
                            qty=recommend_quantity(limit_price),
                            time_in_force='day',
                            order_class='bracket',
                            limit_price=limit_price,
                            take_profit=dict(
                                limit_price=limit_price + (candle_range * 3),
                            ),
                            stop_loss=dict(
                                stop_price=limit_price - current_atr,  # Trailing stop loss using ATR
                                trail_price=current_atr * 0.5  # Adjust as needed
                            )
                        )
                    else:
                        print(f"Already ordered for {symbol}, skipping")

                # Short Entry Condition
                elif current_candle.close < upper[-1] and previous_candle.close > upper[-2]:
                    print(f"{symbol} closed below upper Bollinger Band")

                    if symbol not in existing_order_symbols:
                        limit_price = current_candle.close
                        candle_range = current_candle.high - current_candle.low

                        messages.append(f"Placing short order for {symbol} at {limit_price} \n\n")

                        print(f"Placing short order for {symbol} at {limit_price}")

                        trading_client.submit_order(
                            symbol=symbol,
                            side='sell',
                            type='limit',
                            qty=recommend_quantity(limit_price),
                            time_in_force='day',
                            order_class='bracket',
                            limit_price=limit_price,
                            take_profit=dict(
                                limit_price=limit_price - (candle_range * 3),
                            ),
                            stop_loss=dict(
                                stop_price=limit_price + current_atr,  # Trailing stop loss using ATR
                                trail_price=current_atr * 0.5  # Adjust as needed
                            )
                        )
                    else:
                        print(f"Already ordered for {symbol}, skipping")

    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")

#Send email notification
with smtplib.SMTP_SSL(config.EMAIL_HOST, config.EMAIL_PORT, context=context) as server:
    server.login(config.EMAIL_ADDRESS, config.EMAIL_PASSWORD)

    email_message = f"Subject: Trade Notifications for {today}\n\n"
    email_message += "\n\n".join(messages)

    server.sendmail(config.EMAIL_ADDRESS, config.EMAIL_ADDRESS, email_message)   