import sqlite3
import config
import datetime
import numpy as np
import pandas as pd
import smtplib
import ssl
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus
from helpers import recommend_quantity
import tulipy as ti

def nadaraya_watson_smoothing(prices, bandwidth):
    n = len(prices)
    smoothed = np.zeros(n)
    for i in range(n):
        weights = np.exp(-0.5 * ((np.arange(n) - i) / bandwidth) ** 2)
        weights /= weights.sum()
        smoothed[i] = np.dot(weights, prices)
    return smoothed

context = ssl.create_default_context()

# DB Connection
connection = sqlite3.connect(config.DB_PATH)
connection.row_factory = sqlite3.Row
cursor = connection.cursor()

# Strategy: nadaraya_watson_envelope
cursor.execute("""SELECT id FROM strategy WHERE name = "nadarya_watson" """)
strategy_id = cursor.fetchone()['id']

cursor.execute("""
    SELECT symbol FROM stock
    JOIN stock_strategy ON stock_strategy.stock_id = stock.id
    WHERE stock_strategy.strategy_id = ?
""", (strategy_id,))
stocks = cursor.fetchall()

symbols = [stock['symbol'] for stock in stocks]

# Alpaca Init
trading_client = TradingClient(config.API_KEY, config.SECRET_KEY, paper=True)
data_client = StockHistoricalDataClient(config.API_KEY, config.SECRET_KEY)

# Date Setup
current_date = "2025-04-12"
today = datetime.datetime.strptime(current_date, "%Y-%m-%d").strftime("%Y-%m-%d")

orders_request = GetOrdersRequest(
    status=QueryOrderStatus.ALL,
    limit=500,
    after=today
)

orders = trading_client.get_orders(filter=orders_request)
existing_order_symbols = [order.symbol for order in orders if order.status != 'canceled']

start_minute_bar = f"{today}T09:30:00"
end_minute_bar = f"{today}T16:00:00"

messages = []

for symbol in symbols:
    try:
        request_params = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute,
            start=f"{today}T09:30:00Z",
            end=f"{today}T16:00:00Z",
            feed="iex"
        )
        bars = data_client.get_stock_bars(request_params).df

        if symbol in bars.index:
            bars = bars.loc[symbol]

            market_open_mask = (bars.index >= start_minute_bar) & (bars.index < end_minute_bar)
            market_open_bars = bars.loc[market_open_mask]

            if len(market_open_bars) >= 30:
                closes = market_open_bars.close.values
                highs = market_open_bars.high.values
                lows = market_open_bars.low.values

                bandwidth = 8
                envelope_multiplier = 3

                smoothed = nadaraya_watson_smoothing(closes, bandwidth)
                rolling_std = pd.Series(closes).rolling(window=bandwidth, min_periods=1).std().values

                upper_band = smoothed + envelope_multiplier * rolling_std
                lower_band = smoothed - envelope_multiplier * rolling_std

                current_candle = market_open_bars.iloc[-1]
                previous_candle = market_open_bars.iloc[-2]

                atr_values = ti.atr(highs, lows, closes, period=14)
                current_atr = atr_values[-1]

                # Long Entry
                if current_candle.close > lower_band[-1] and previous_candle.close < lower_band[-2]:
                    if symbol not in existing_order_symbols:
                        limit_price = current_candle.close
                        candle_range = current_candle.high - current_candle.low

                        messages.append(f"Placing long order for {symbol} at {limit_price}")

                        trading_client.submit_order(
                            symbol=symbol,
                            side='buy',
                            type='market',  # changed here
                            qty=recommend_quantity(current_candle.close),
                            time_in_force='day',
                            order_class='bracket',
                            take_profit=dict(
                                limit_price=current_candle.close + (candle_range * 3),
                            ),
                            stop_loss=dict(
                                stop_price=current_candle.close - current_atr,
                                trail_price=current_atr * 0.5
                            )
                        )


                # Short Entry
                elif current_candle.close < upper_band[-1] and previous_candle.close > upper_band[-2]:
                    if symbol not in existing_order_symbols:
                        limit_price = current_candle.close
                        candle_range = current_candle.high - current_candle.low

                        messages.append(f"Placing short order for {symbol} at {limit_price}")

                        trading_client.submit_order(
                            symbol=symbol,
                            side='buy',
                            type='market',  # changed here
                            qty=recommend_quantity(current_candle.close),
                            time_in_force='day',
                            order_class='bracket',
                            take_profit=dict(
                                limit_price=current_candle.close - (candle_range * 3),
                            ),
                            stop_loss=dict(
                                stop_price=current_candle.close + current_atr,
                                trail_price=current_atr * 0.5
                            )
                        )


    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")


print(messages)
# Email Notification
# with smtplib.SMTP_SSL(config.EMAIL_HOST, config.EMAIL_PORT, context=context) as server:
#     server.login(config.EMAIL_ADDRESS, config.EMAIL_PASSWORD)

#     email_message = f"Subject: Trade Notifications for {today}\n\n"
#     email_message += "\n\n".join(messages)

#     server.sendmail(config.EMAIL_ADDRESS, config.EMAIL_ADDRESS, email_message) 
