import sqlite3, config
import bcrypt,datetime, random, hashlib, requests
from fastapi import FastAPI, Request, Form, Query, Depends, HTTPException
from fastapi.templating import Jinja2Templates 
from fastapi.responses import RedirectResponse, JSONResponse, Response
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus
import json
from starlette.middleware.sessions import SessionMiddleware
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from typing import List,Dict, Any 
from pydantic import BaseModel
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from itertools import product
from sklearn.metrics import mean_squared_error, precision_score, recall_score, accuracy_score
import tulipy as ti



user_id = "1"

app = FastAPI()
templates = Jinja2Templates(directory="templates")

app.add_middleware(SessionMiddleware, secret_key=config.SPECIAL_KEY)

app.mount("/static", StaticFiles(directory="static"), name="static")

def get_db_connection():
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@app.get("/")
def root():
    return RedirectResponse(url="/login")

@app.get("/login")
def login(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/signup")
def signup(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})

@app.post("/signup")
def signup(request: Request, username: str = Form(...), email: str = Form(...), password: str = Form(...)):
    connection = get_db_connection()
    cursor = connection.cursor()

    # Check if email already exists
    cursor.execute("SELECT * FROM user WHERE email = ?", (email,))
    existing_user = cursor.fetchone()

    if existing_user:
        return templates.TemplateResponse("signup.html", {"request": request, "error": "Email is already registered, please login"})


    # Hash the password before storing
    hashed_password = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())

    cursor.execute("INSERT INTO user (username, email, password_hash) VALUES (?, ?, ?)",(username, email, hashed_password.decode("utf-8")))

    connection.commit()
    connection.close()

    return RedirectResponse(url="/portfolio_dashboard", status_code=303)

@app.post("/login")
def login(request: Request,email: str = Form(...), password: str = Form(...)):
    connection = get_db_connection()
    cursor = connection.cursor()

    cursor.execute("SELECT id, username, password_hash FROM user WHERE email = ?", (email,))
    user = cursor.fetchone()

    if not user:
        return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid Email"})

    user_id, username, stored_password_hash = user

    # Verify password
    if not bcrypt.checkpw(password.encode("utf-8"), stored_password_hash.encode("utf-8")):
        return templates.TemplateResponse("login.html", {"request": request,"error": "Invalid Password"})
    
    request.session["user_id"] = user["id"]
    
    connection.close()
    
    return RedirectResponse(url="/portfolio_dashboard", status_code=303)

@app.get("/get_current_user")
def get_current_user(request: Request):
    user_id = request.session.get("user_id")
    
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    return {"user_id": user_id}

@app.get("/index")
def index(request: Request):

    stock_filter = request.query_params.get('filter', False)
    connection = get_db_connection()

    cursor = connection.cursor()
    
    if stock_filter == 'new_closing_highs':
        cursor.execute("""
                    SELECT * FROM (
                       select symbol, name, stock_id, max(close), date
                       from stock_price join stock on stock.id = stock_price.stock_id
                       group by stock_id
                       order by symbol
                       ) where date = (select max(date) from stock_price)
                """)
    elif stock_filter == 'new_closing_lows':
        cursor.execute("""
                    SELECT * FROM (
                       select symbol, name, stock_id, min(close), date
                       from stock_price join stock on stock.id = stock_price.stock_id
                       group by stock_id
                       order by symbol
                       ) where date = (select max(date) from stock_price)
                """)
    elif stock_filter == 'rsi_overbought':
        cursor.execute("""
                       select symbol, name, stock_id, date
                       from stock_price join stock on stock.id = stock_price.stock_id
                       where rsi_14 > 70 AND date = (select max(date) from stock_price) 
                       order by symbol
                """)
    elif stock_filter == 'rsi_oversold':
        cursor.execute("""
                    select symbol, name, stock_id, date
                       from stock_price join stock on stock.id = stock_price.stock_id
                       where rsi_14 < 30 AND date = (select max(date) from stock_price) 
                       order by symbol
                """)
    elif stock_filter == 'above_sma_20':
        cursor.execute("""
                    select symbol, name, stock_id, date
                       from stock_price join stock on stock.id = stock_price.stock_id
                       where close > sma_20 AND date = (select max(date) from stock_price) 
                       order by symbol
                """)
    elif stock_filter == 'below_sma_20':
        cursor.execute("""
                    select symbol, name, stock_id, date
                       from stock_price join stock on stock.id = stock_price.stock_id
                       where close < sma_20 AND date = (select max(date) from stock_price) 
                       order by symbol
                """)
    elif stock_filter == 'above_sma_50':
        cursor.execute("""
                    select symbol, name, stock_id, date
                       from stock_price join stock on stock.id = stock_price.stock_id
                       where close > sma_50 AND date = (select max(date) from stock_price) 
                       order by symbol
                """)
    elif stock_filter == 'below_sma_50':
        cursor.execute("""
                    select symbol, name, stock_id, date
                       from stock_price join stock on stock.id = stock_price.stock_id
                       where close < sma_50 AND date = (select max(date) from stock_price) 
                       order by symbol
                """)
    elif stock_filter == 'weekly_change_5_percent':
        cursor.execute("""
            SELECT s.symbol, s.name, sp.stock_id, sp.date, 
                ROUND(((sp.close - prev_sp.close) / prev_sp.close) * 100, 2) AS weekly_change
            FROM stock_price sp
            JOIN stock s ON s.id = sp.stock_id
            JOIN stock_price prev_sp 
                ON prev_sp.stock_id = sp.stock_id 
                AND prev_sp.date = DATE(sp.date, '-7 days')
            WHERE sp.date = (SELECT MAX(date) FROM stock_price)
            AND ROUND(((sp.close - prev_sp.close) / prev_sp.close) * 100, 2) >= 10
            ORDER BY s.symbol
        """)

    else:
        cursor.execute("""
                        SELECT symbol, name FROM stock ORDER BY symbol
                    """)

    rows = cursor.fetchall()

    cursor.execute("""
        select symbol, rsi_14, sma_20, sma_50, close from  stock join stock_price on stock_price.stock_id = stock.id where date = (select max(date) from stock_price);
    """)

    indicator_rows = cursor.fetchall()
    indicator_values = {}

    for row in indicator_rows:
        indicator_values[row['symbol']] = row

    return templates.TemplateResponse("index.html", {"request": request, "stocks": rows, "indicator_values" : indicator_values})
    
@app.get("/search_stocks")
def search_stocks(query: str = Query("")):
    connection = get_db_connection()
    cursor = connection.cursor()
    # Use SQL wildcard '%' to match stocks starting with the query
    cursor.execute("""
        SELECT symbol, name FROM stock
        WHERE symbol LIKE ? 
        ORDER BY symbol
    """, (query + '%',))  # Match stocks that start with 'query'

    rows = cursor.fetchall()
    return JSONResponse([{"symbol": row["symbol"], "name": row["name"]} for row in rows])

@app.get("/stock/{symbol}")
def stock_details(request: Request, symbol):

    connection = get_db_connection()

    cursor = connection.cursor()

    cursor.execute("""
                    SELECT * FROM strategy 
                """)
    
    strategies = cursor.fetchall()

    cursor.execute("""
                    SELECT * FROM stock WHERE symbol = ?
                """, (symbol,))

    row = cursor.fetchone()

    cursor.execute("""
                    SELECT * FROM stock_price WHERE stock_id = ? ORDER BY date DESC
                """, (row['id'],))
    
    prices = cursor.fetchall()

    return templates.TemplateResponse("stock_details.html", {"request": request, "stock": row, "bars":prices, "strategies":strategies})

@app.post("/apply_strategy")
def apply_strategy(strategy_id: int = Form(...), stock_id: int = Form(...)):
    connection = get_db_connection()
    cursor = connection.cursor()

    # 🔹 Check if strategy is already applied to stock
    cursor.execute("""
        SELECT 1 FROM stock_strategy WHERE stock_id = ? AND strategy_id = ?
    """, (stock_id, strategy_id))
    
    existing_entry = cursor.fetchone()

    if not existing_entry:
        # ✅ Only insert if the pair does not exist
        cursor.execute("""
            INSERT INTO stock_strategy (stock_id, strategy_id) VALUES (?, ?)
        """, (stock_id, strategy_id))
        connection.commit()

    return RedirectResponse(url=f"/strategy/{strategy_id}", status_code=303)

@app.get("/strategies")
def strategies(request:Request):

    connection = get_db_connection()
    cursor = connection.cursor()

    cursor.execute("""
        select * from strategy
    """)

    strategies = cursor.fetchall()

    return templates.TemplateResponse("strategies.html",{"request":request, "strategies" : strategies})  

@app.get("/orders")
def orders(request:Request):

    trading_client = TradingClient(config.API_KEY, config.SECRET_KEY, paper=True)

    orders_request = GetOrdersRequest(status=QueryOrderStatus.ALL)
    orders = trading_client.get_orders(filter=orders_request)

    return templates.TemplateResponse("orders.html",{"request":request, "orders" : orders})  
    
@app.get("/strategy/{strategy_id}")
def strategy(request:Request, strategy_id):

    connection = get_db_connection()
    cursor = connection.cursor()

    cursor.execute("""
                    SELECT * FROM strategy WHERE id = ?
                """, (strategy_id,)) 

    strategy = cursor.fetchone()

    cursor.execute("""
                    SELECT symbol,name FROM stock JOIN stock_strategy on stock_strategy.stock_id = stock.id WHERE strategy_id = ?
                """, (strategy_id,)) 

    stocks = cursor.fetchall()

    return templates.TemplateResponse("strategy.html",{"request":request,"stocks":stocks,"strategy":strategy}) 

@app.get("/trade_stocks")
def trade_stocks(request:Request):

    return templates.TemplateResponse("trade_stocks.html",{"request":request})

@app.get("/get_stock_price")
def get_stock_price(symbol: str):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT sp.close 
        FROM stock_price sp
        JOIN stock s ON sp.stock_id = s.id
        WHERE s.symbol = ?
        ORDER BY sp.date DESC
        LIMIT 1
    """, (symbol.upper(),))
    
    row = cursor.fetchone()
    conn.close()

    if row:
        return {"price": row["close"]}
    else:
        raise HTTPException(status_code=404, detail="Stock symbol not found")
    
@app.route("/buy_stock", methods=["GET", "POST"])
async def buy_stock(
    request: Request
):
    # user_id = request.session.get("user_id")  # ✅ Extract user_id from session
    # if not user_id:
    #     raise HTTPException(status_code=401, detail="Not authenticated")

    # Extract form data
    form_data = await request.form()
    symbol = form_data.get("symbol")
    shares = form_data.get("shares")
    price = form_data.get("price")

    if not symbol or not shares or not price:
        return JSONResponse(content={"error": "Missing required fields"}, status_code=400)

    try:
        shares = int(shares)  # Convert to int
        price = float(price)  # Convert to float
    except ValueError:
        return JSONResponse(content={"error": "Invalid input for shares or price"}, status_code=400)

    conn = get_db_connection()
    cursor = conn.cursor()

    # Get user cash balance
    cursor.execute("SELECT cash FROM user WHERE id = ?", (user_id,))
    user = cursor.fetchone()

    if not user:
        return JSONResponse(content={"error": "User not found"}, status_code=400)

    cash_balance = user["cash"]
    total_cost = shares * price

    if cash_balance < total_cost:
        return JSONResponse(content={"error": "Insufficient funds"}, status_code=400)

    # Deduct cash from user balance
    cursor.execute("UPDATE user SET cash = cash - ? WHERE id = ?", (total_cost, user_id))

    # Check if user already holds the stock
    cursor.execute("SELECT * FROM stock_holding WHERE user_id = ? AND company_symbol = ?", (user_id, symbol))
    holding = cursor.fetchone()

    if holding:
        # Update existing holding
        new_shares = holding["number_of_shares"] + shares
        new_investment = holding["investment_amount"] + total_cost

        # Update buying values JSON
        buying_value = json.loads(holding["buying_value"])
        buying_value.append({"shares": shares, "price": price})

        cursor.execute("""
            UPDATE stock_holding 
            SET number_of_shares = ?, investment_amount = ?, buying_value = ?
            WHERE user_id = ? AND company_symbol = ?
        """, (new_shares, new_investment, json.dumps(buying_value), user_id, symbol))
    else:
        # Insert new holding with sector
        cursor.execute("""
            INSERT INTO stock_holding (user_id, company_symbol, number_of_shares, investment_amount, buying_value)
            VALUES (?, ?, ?, ?, ?)
        """, (user_id, symbol, shares, total_cost, json.dumps([{"shares": shares, "price": price}])))

    # Record transaction
    cursor.execute("""
        INSERT INTO transactions (user_id, symbol, shares, price, transaction_type)
        VALUES (?, ?, ?, ?, 'BUY')
    """, (user_id, symbol, shares, price))

    conn.commit()
    conn.close()

    return JSONResponse(content={"message": f"Stock {symbol} purchased successfully!"}, status_code=200)

@app.route("/sell_stock", methods=["POST"])
async def sell_stock(request: Request): 
    
    # user_id = request.session.get("user_id")
    # if not user_id:
    #     raise HTTPException(status_code=401, detail="Not authenticated")
    
    conn = get_db_connection()
    cursor = conn.cursor()

    form_data = await request.form()
    symbol = form_data.get("symbol")
    shares = form_data.get("shares")
    price = form_data.get("price")

    try:
        shares = int(shares)
        price = float(price)
    except ValueError:
        return JSONResponse(content={"error": "Invalid input for shares or price"}, status_code=400)

    # Get user stock holdings
    cursor.execute("SELECT * FROM stock_holding WHERE user_id = ? AND company_symbol = ?", (user_id, symbol))
    holding = cursor.fetchone()

    if not holding or holding["number_of_shares"] < shares:
        return JSONResponse(content={"error": "Not enough shares to sell"}, status_code=400)

    # Get all past buy transactions (LIFO order)
    cursor.execute("""
        SELECT id, shares, price FROM transactions 
        WHERE user_id = ? AND symbol = ? AND transaction_type = 'BUY'
        ORDER BY id DESC  -- LIFO (latest buy first)
    """, (user_id, symbol))
    
    buy_transactions = cursor.fetchall()

    qty_to_sell = shares
    total_profit_loss = 0

    while qty_to_sell > 0 and buy_transactions:
        last_buy = buy_transactions.pop(0)  # Get latest buy (LIFO)
        buy_id, buy_shares, buy_price = last_buy

        if buy_shares <= qty_to_sell:
            # Fully sell this batch
            profit_loss = (price - buy_price) * buy_shares
            total_profit_loss += profit_loss
            qty_to_sell -= buy_shares
            # Delete this buy record since all shares are sold
            # cursor.execute("DELETE FROM transactions WHERE id = ?", (buy_id,))
        else:
            # Partially sell this batch
            profit_loss = (price - buy_price) * qty_to_sell
            total_profit_loss += profit_loss
            new_buy_shares = buy_shares - qty_to_sell
            qty_to_sell = 0
            # Update remaining shares in this buy transaction
            # cursor.execute("UPDATE transactions SET shares = ? WHERE id = ?", (new_buy_shares, buy_id))

    # Update stock holdings
    new_shares = holding["number_of_shares"] - shares
    new_cash = shares * price  

    if new_shares == 0:
        cursor.execute("DELETE FROM stock_holding WHERE user_id = ? AND company_symbol = ?", (user_id, symbol))
    else:
        cursor.execute("""
            UPDATE stock_holding
            SET number_of_shares = ?, investment_amount = investment_amount - ?
            WHERE user_id = ? AND company_symbol = ?
        """, (new_shares, new_cash, user_id, symbol))

    # Add cash to user balance
    cursor.execute("UPDATE user SET cash = cash + ? WHERE id = ?", (new_cash, user_id))

    # Record sell transaction WITH profit/loss
    cursor.execute("""
        INSERT INTO transactions (user_id, symbol, shares, price, transaction_type, profit_loss)
        VALUES (?, ?, ?, ?, 'SELL', ?)
    """, (user_id, symbol, shares, price, round(total_profit_loss, 2)))

    conn.commit()
    conn.close()

    return JSONResponse(content={
        "message": "Stock sold successfully!",
        "profit_loss": round(total_profit_loss, 2)
    }, status_code=200)


@app.get("/portfolio_dashboard")
def get_portfolio_dashboard(request: Request):

    conn = get_db_connection()
    cursor = conn.cursor()

      # Replace with session-based user ID

    # ✅ Fetch Cash Balance
    cursor.execute("SELECT cash FROM user WHERE id = ?", (user_id,))
    cash_balance_row = cursor.fetchone()
    cash_balance = cash_balance_row["cash"] if cash_balance_row else 0

    # ✅ Fetch Holdings
    cursor.execute("""
        SELECT sh.company_symbol, sh.number_of_shares, sh.investment_amount, s.name, s.id AS stock_id
        FROM stock_holding sh
        JOIN stock s ON sh.company_symbol = s.symbol
        WHERE sh.user_id = ?
    """, (user_id,))
    
    holdings = cursor.fetchall()
    
    if not holdings:
        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "holdings": [],
            "totalInvestment": 0,
            "cashBalance": cash_balance,
            "totalPortfolioValue": cash_balance,
            "dailyChangePercent": 0,
            "CAGR": 0,
            "MaxDrawdown": 0,
            "SharpeRatio": 0,
            "WinRate": 0
        })

    portfolio_data = []
    total_value = 0
    total_investment = 0
    portfolio_returns = []  # For Sharpe Ratio
    equity_curve = []  # For Max Drawdown
    total_wins = 0
    total_trades = 0

    for holding in holdings:
        symbol = holding["company_symbol"]
        shares = holding["number_of_shares"]
        investment = holding["investment_amount"]
        name = holding["name"]
        stock_id = holding["stock_id"]

        # ✅ Fetch latest stock price
        cursor.execute("""
            SELECT close 
            FROM stock_price 
            WHERE stock_id = ? 
            ORDER BY date DESC 
            LIMIT 1
        """, (stock_id,))
        latest_price_row = cursor.fetchone()
        latest_price = latest_price_row["close"] if latest_price_row else 0

        # ✅ Fetch Previous Close Price
        cursor.execute("""
            SELECT close 
            FROM stock_price 
            WHERE stock_id = ? 
            ORDER BY date DESC 
            LIMIT 1 OFFSET 1
        """, (stock_id,))
        prev_close_row = cursor.fetchone()
        prev_close = prev_close_row["close"] if prev_close_row else latest_price

        # ✅ Calculate market value, P&L, and daily change %
        market_value = latest_price * shares
        unrealized_pnl = market_value - investment
        unrealized_pnl_percent = (unrealized_pnl / investment) * 100 if investment > 0 else 0
        daily_change_percent = ((latest_price - prev_close) / prev_close) * 100 if prev_close > 0 else 0

        total_value += market_value
        total_investment += investment

        # ✅ Store data for performance metrics
        portfolio_returns.append(unrealized_pnl_percent)
        equity_curve.append(total_value + cash_balance)  # Track portfolio value over time

        portfolio_data.append({
            "CompanySymbol": symbol,
            "CompanyName": name,
            "NumberShares": shares,
            "InvestmentAmount": investment,
            "AverageCost": investment / shares if shares > 0 else 0,
            "MarketValue": market_value,
            "UnrealizedPNL": unrealized_pnl,
            "UnrealizedPNLPercent": unrealized_pnl_percent,
            "DailyChangePercent": daily_change_percent
        })

    # ✅ Compute Final Portfolio Metrics
    total_portfolio_value = total_value + cash_balance
    total_unrealized_pnl = total_value - total_investment
    total_unrealized_pnl_percent = (total_unrealized_pnl / total_investment) * 100 if total_investment > 0 else 0

    # ✅ Fetch all trades for the user
    cursor.execute("""
        SELECT symbol, price, shares, transaction_type, timestamp
        FROM transactions 
        WHERE user_id = ?
        ORDER BY timestamp ASC
    """, (user_id,))
    
    transactions = cursor.fetchall()

    if transactions:
        first_trade_date = transactions[0]["timestamp"][:10]  # Get YYYY-MM-DD
        last_trade_date = transactions[-1]["timestamp"][:10]  # Get YYYY-MM-DD

        years_held = (datetime.datetime.strptime(last_trade_date, "%Y-%m-%d") - datetime.datetime.strptime(first_trade_date, "%Y-%m-%d")).days / 365.0

        if total_investment <= 0 or years_held <= 0:
            CAGR = 0  # Avoid division by zero
        else:
            try:
                CAGR = ((total_portfolio_value / total_investment) ** (1 / years_held) - 1) * 100
            except OverflowError:
                CAGR = float('inf')  # Assign infinity if value is too large

        # ✅ Calculate Win Rate
        for trade in transactions:
            if trade["transaction_type"] in ["SELL", "COVER"]:  # Only count closing trades
                profit_or_loss = (trade["price"] * trade["shares"]) - total_investment
                if profit_or_loss > 0:
                    total_wins += 1
                total_trades += 1

        WinRate = (total_wins / total_trades) * 100 if total_trades > 0 else 0

    else:
        CAGR = 0
        WinRate = 0

    # ✅ Calculate Max Drawdown
    peak = equity_curve[0]
    max_drawdown = 0
    for value in equity_curve:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        max_drawdown = max(max_drawdown, drawdown)

    MaxDrawdown = max_drawdown * 100

    # ✅ Calculate Sharpe Ratio
    portfolio_std_dev = np.std(portfolio_returns)
    risk_free_rate = 2.0  # Assuming 2% risk-free return
    SharpeRatio = ((np.mean(portfolio_returns) - risk_free_rate) / portfolio_std_dev) if portfolio_std_dev > 0 else 0

    # Categorize holdings into Stocks, ETFs, and Index Funds
    asset_counts = {"Stocks": 0, "ETFs": 0, "Index Funds": 0}

    for holding in portfolio_data:
        name = holding["CompanyName"].lower()
        
        if "etf" in name:
            asset_counts["ETFs"] += 1
        elif "index" in name or "fund" in name:
            asset_counts["Index Funds"] += 1
        else:
            asset_counts["Stocks"] += 1

    context = {
        "request": request,
        "holdings": portfolio_data,
        "totalInvestment": total_investment,
        "cashBalance": cash_balance,
        "totalPortfolioValue": total_portfolio_value,
        "totalUnrealizedPNL": total_unrealized_pnl,
        "totalUnrealizedPNLPercent": total_unrealized_pnl_percent,
        "dailyChangePercent": ((total_value - total_investment) / total_investment) * 100 if total_investment > 0 else 0,
        "CAGR": CAGR,
        "MaxDrawdown": MaxDrawdown,
        "SharpeRatio": SharpeRatio,
        "WinRate": WinRate,
        "assetCounts": asset_counts, 
        "MarketValue" : market_value, # New Data for Asset Type Chart
    }

    conn.close()

    return templates.TemplateResponse("dashboard.html", context)


@app.route("/transaction_history", methods=["GET"])
def transaction_history(request: Request):

    # user_id = request.session.get("user_id")
    # if not user_id:
    #     raise HTTPException(status_code=401, detail="Not authenticated")

    conn = get_db_connection()
    cursor = conn.cursor()

    # Fetch all transactions (buys & sells)
    cursor.execute("""
        SELECT timestamp, symbol, shares, price, transaction_type, 
        COALESCE(profit_loss, 0) as profit_loss  -- ✅ Ensure no NULL values
        FROM transactions WHERE user_id = ? 
        ORDER BY timestamp DESC
    """, (user_id,))
    
    transactions = cursor.fetchall()
    
    # Convert to list of dictionaries
    transaction_list = []
    for t in transactions:
        transaction_data = {
            "timestamp": t["timestamp"],
            "symbol": t["symbol"],
            "shares": t["shares"],
            "price": t["price"],
            "transaction_type": t["transaction_type"],
            "profit_loss": t["profit_loss"] if t["transaction_type"] == "SELL" else None  # ✅ Always include it
        }
        transaction_list.append(transaction_data)


    conn.close()
    return templates.TemplateResponse("transaction_history.html", {
        "request": request, 
        "transactions": transaction_list
    })

def get_all_stock_recommendations():
    """Fetch all stocks and generate recommendations."""
    conn = sqlite3.connect(config.DB_PATH)
    query = """
        SELECT s.symbol, s.name, sp.close
        FROM stock_price sp
        JOIN stock s ON s.id = sp.stock_id
        WHERE sp.date = (SELECT MAX(date) FROM stock_price WHERE stock_id = s.id);
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        print("No stock data found in database.")
        return []

    # Generate technical indicators
    df["Returns"] = df["close"].pct_change()
    df["MA_50"] = df["close"].rolling(window=50).mean()
    df["MA_200"] = df["close"].rolling(window=200).mean()
    df["Volatility"] = df["Returns"].rolling(window=50).std()

    # Drop rows with NaN values in critical columns (ensuring proper data for K-Means)
    df.dropna(subset=["Returns", "MA_50", "MA_200", "Volatility"], inplace=True)

    if df.empty:
        print("No valid data after dropping NaN values.")
        return []

    # Normalize the features
    feature_cols = ["Returns", "MA_50", "MA_200", "Volatility"]
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # Apply K-Means clustering with 9 clusters
    kmeans = KMeans(n_clusters=9, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(df[feature_cols])

    # Debugging: Check cluster distribution
    print("Cluster distribution:", df["Cluster"].value_counts())

    # Define recommendation mapping for 9 clusters
    cluster_map = {
        0: "Strong Buy",
        1: "Buy",
        2: "Weak Buy",
        3: "Hold",
        4: "Weak Sell",
        5: "Sell",
        6: "Strong Sell",
        7: "Watchlist",
        8: "Volatile"
    }

    # Map clusters to recommendations
    df["recommendation"] = df["Cluster"].map(cluster_map)


    return df[["symbol", "close", "recommendation"]].to_dict(orient="records")


def save_signals(signals):
    """Save buy/sell signals into the database."""
    conn = sqlite3.connect(config.DB_PATH)
    cursor = conn.cursor()


    if signals:
        cursor.executemany(
            "INSERT INTO trade_signals (stock_id, signal, indicator) VALUES (?, ?, ?)",
            signals
        )
        conn.commit()

    conn.close()

def get_indicator_recommendations():
    """Fetch stock prices, compute indicators, generate recommendations, and save signals."""
    
    conn = sqlite3.connect(config.DB_PATH)
    query = """
        SELECT s.symbol, sp.stock_id, sp.close
        FROM stock_price sp
        JOIN stock s ON s.id = sp.stock_id
        WHERE sp.date = (SELECT MAX(date) FROM stock_price WHERE stock_id = s.id);
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        print("No stock data found in database.")
        return []

    recommendations = []
    trade_signals = []

    for _, row in df.iterrows():
        stock_id = row["stock_id"]
        symbol = row["symbol"]

        # Fetch historical close prices for indicator calculation
        conn = sqlite3.connect(config.DB_PATH)
        historical_query = f"""
            SELECT close FROM stock_price WHERE stock_id = {stock_id} ORDER BY date ASC
        """
        prices_df = pd.read_sql_query(historical_query, conn)
        conn.close()

        prices = prices_df["close"].values
        if len(prices) < 26:  # Minimum data needed for MACD
            continue

        prices_np = np.array(prices, dtype=np.float64)
        recommendation = "Hold"
        indicator_used = "None"

        # Bollinger Bands (20-period, 2 std dev)
        if len(prices_np) >= 20:
            upper, middle, lower = ti.bbands(prices_np, period=20, stddev=2)
            current_price = prices_np[-1]
            previous_price = prices_np[-2]

            if current_price > lower[-1] and previous_price < lower[-2]:
                recommendation = "Buy"
                indicator_used = "Bollinger Bands"
            elif current_price < upper[-1] and previous_price > upper[-2]:
                recommendation = "Sell"
                indicator_used = "Bollinger Bands"


        # MACD (12, 26, 9) - Overwrites previous recommendation if triggered
        if len(prices_np) >= 26:
            macd, signal, hist = ti.macd(prices_np, short_period=12, long_period=26, signal_period=9)
            if macd[-1] > signal[-1] and macd[-2] <= signal[-2]:  # MACD crosses above Signal
                recommendation = "Buy"
                indicator_used = "MACD"
            elif macd[-1] < signal[-1] and macd[-2] >= signal[-2]:  # MACD crosses below Signal
                recommendation = "Sell"
                indicator_used = "MACD"

        recommendations.append({
            "symbol": symbol,
            "close": row["close"],
            "recommendation": recommendation,
            "indicator_used": indicator_used  # Add indicator name
        })


        if recommendation != "Hold":
            trade_signals.append((stock_id, recommendation, indicator_used))

    # Save trade signals to database
    # save_signals(trade_signals)

    return recommendations


@app.get("/recommendations/")
def stock_recommendations_page(request: Request):
    recommendations_kmeans = get_all_stock_recommendations()
    recommendations_indicators = get_indicator_recommendations()
    return templates.TemplateResponse("recommendations.html", {"request": request, "recommendations_kmeans": recommendations_kmeans,"recommendations_indicators" : recommendations_indicators})


@app.get("/logout")
def logout(request: Request,user_id: int = Depends(get_current_user)):
    request.session.clear()
    return RedirectResponse(url="/login", status_code=303)

@app.route("/profit_loss_chart")
def profit_loss_chart_data(request: Request):
    user_id = request.session.get("user_id")

    conn = get_db_connection()
    cursor = conn.cursor()

    # Fetch all SELL transactions with profit/loss
    cursor.execute("""
        SELECT timestamp, profit_loss FROM transactions 
        WHERE user_id = ? AND transaction_type = 'SELL'
        ORDER BY timestamp ASC
    """, (user_id,))
    
    transactions = cursor.fetchall()
    conn.close()


    # Convert to JSON format
    chart_data = {
        "timestamps": [t["timestamp"] for t in transactions],
        "profit_losses": [t["profit_loss"] for t in transactions]
    }

    return JSONResponse(content=chart_data)

@app.get("/predictions")
def trade_stocks(request:Request):

    return templates.TemplateResponse("predictions.html",{"request":request})

def ARIMA_ALGO(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    uniqueVals = df['symbol'].unique()
    df = df.set_index('symbol')

    def parser(x: str) -> datetime.datetime:
        return datetime.datetime.strptime(x, '%Y-%m-%d')

    def arima_model(train, test):
        history = [x for x in train]
        predictions = []
        for t in range(len(test)):
            model = ARIMA(history, order=(6, 1, 0))
            model_fit = model.fit()
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            history.append(test[t])
        return predictions

    results = {}
    for company in uniqueVals:
        data = df.loc[company].reset_index()
        data['Price'] = data['close']
        Quantity_date = data[['Price', 'date']]
        Quantity_date.index = Quantity_date['date'].map(parser)
        Quantity_date['Price'] = Quantity_date['Price'].astype(float)
        Quantity_date = Quantity_date.bfill().drop(columns=['date'])

        quantity = Quantity_date.values
        size = int(len(quantity) * 0.80)
        train, test = quantity[:size], quantity[size:]

        predictions = arima_model(train, test)

        actual_directions = [1 if test[i, 0] > test[i-1, 0] else 0 for i in range(1, len(test))]
        predicted_directions = [1 if predictions[i] > test[i-1, 0] else 0 for i in range(1, len(predictions))]

        correct_predictions = sum([1 for i in range(len(actual_directions)) 
                                   if actual_directions[i] == predicted_directions[i]])
        directional_accuracy = (correct_predictions / len(actual_directions)) * 100
        
        # Precision, Recall, and Accuracy Calculation
        precision = precision_score(actual_directions, predicted_directions)
        recall = recall_score(actual_directions, predicted_directions)
        accuracy = accuracy_score(actual_directions, predicted_directions)

        # Plot graph
        fig = plt.figure(figsize=(7.2, 4.8), dpi=65)
        plt.plot(test[:, 0], label='Actual Price')
        plt.plot(predictions, label='Predicted Price')
        plt.legend(loc=4)
        plt.savefig(f'static/graph/{company}_ARIMA.png')
        plt.close(fig)

        error_arima = math.sqrt(mean_squared_error(test[:, 0], predictions))
        arima_pred = predictions[-2]
        previous_close = test[-2, 0]

        results[company] = {
            'prediction': arima_pred,
            'previous_close': round(previous_close, 2),
            'rmse': error_arima,
            'model': 'ARIMA',
            'directional_accuracy': round(directional_accuracy, 2),
            'precision': round(precision, 2),
            'recall': round(recall, 2),
            'accuracy': round(accuracy, 2),
            'timestamp': datetime.datetime.now().timestamp()
        }

    return results


@app.post('/predict/arima')
async def predict_arima(request: Request, stock_symbols: List[str] = Form(...)):
    conn = sqlite3.connect(config.DB_PATH)

    # Handle the case where stock_symbols is a single string with commas
    if len(stock_symbols) == 1 and ',' in stock_symbols[0]:
        stock_symbols_list = [symbol.strip() for symbol in stock_symbols[0].split(',')]
    else:
        stock_symbols_list = [symbol.strip() for symbol in stock_symbols]
    
    query = """
        SELECT s.symbol, sp.date, sp.close
        FROM stock s
        JOIN stock_price sp ON s.id = sp.stock_id
        WHERE s.symbol IN ({})
        ORDER BY sp.date ASC
    """.format(','.join(['?'] * len(stock_symbols_list)))

    df = pd.read_sql_query(query, conn, params=stock_symbols_list)
    conn.close()

    print("Parsed Stock Symbols:", stock_symbols_list) 

    if df.empty:
        raise HTTPException(status_code=404, detail='No stock data found for provided symbols')

    results = ARIMA_ALGO(df)
    print(results)
    return templates.TemplateResponse("predictions.html", {"predictions": results, "request": request})

