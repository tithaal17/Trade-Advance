import sqlite3
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import config

def backtest_recommendations(past_date, future_days):
    conn = sqlite3.connect(config.DB_PATH)
    
    # Fetch stock prices on the given past_date
    query = f'''
        SELECT s.symbol, sp.date, sp.close
        FROM stock_price sp
        JOIN stock s ON s.id = sp.stock_id
        WHERE sp.date = '{past_date}';
    '''
    df = pd.read_sql_query(query, conn)
    
    if df.empty:
        print(f"No stock data found for {past_date}.")
        conn.close()
        return None
    
    # Generate technical indicators
    df["Returns"] = df["close"].pct_change()
    df["MA_50"] = df["close"].rolling(window=50).mean()
    df["MA_200"] = df["close"].rolling(window=200).mean()
    df["Volatility"] = df["Returns"].rolling(window=50).std()
    df.dropna(subset=["Returns", "MA_50", "MA_200", "Volatility"], inplace=True)
    
    # Normalize features
    feature_cols = ["Returns", "MA_50", "MA_200", "Volatility"]
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    # Apply K-Means
    kmeans = KMeans(n_clusters=9, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(df[feature_cols])
    
    # Define recommendations
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
    df["recommendation"] = df["Cluster"].map(cluster_map)
    
    # Get stock prices N days later
    future_date_query = f'''
        SELECT s.symbol, sp.close
        FROM stock_price sp
        JOIN stock s ON s.id = sp.stock_id
        WHERE sp.date = (SELECT MIN(date) FROM stock_price WHERE date > '{past_date}' LIMIT 1 OFFSET 3);
    '''
    future_prices = pd.read_sql_query(future_date_query, conn)
    conn.close()
    
    if future_prices.empty:
        print("No future stock data available.")
        return None
    
    # Merge data
    df = df.merge(future_prices, on="symbol", suffixes=("_past", "_future"))
    df["Price_Change"] = (df["close_future"] - df["close_past"]) / df["close_past"]
    
    # Evaluate accuracy
    df["Correct"] = ((df["recommendation"] == "Strong Buy") & (df["Price_Change"] > 0)) | \
                     ((df["recommendation"] == "Strong Sell") & (df["Price_Change"] < 0))
    
    accuracy = df["Correct"].mean()
    strong_buy_correct = df[(df["recommendation"] == "Strong Buy") & (df["Price_Change"] > 0)].shape[0]
    strong_sell_correct = df[(df["recommendation"] == "Strong Sell") & (df["Price_Change"] < 0)].shape[0]
    total_strong_buy = df[df["recommendation"] == "Strong Buy"].shape[0]
    total_strong_sell = df[df["recommendation"] == "Strong Sell"].shape[0]
    
    print(f"Backtest Accuracy: {accuracy:.2%}")
    print(f"Strong Buy Correct: {strong_buy_correct}/{total_strong_buy}")
    print(f"Strong Sell Correct: {strong_sell_correct}/{total_strong_sell}")
    
    return df[["symbol", "recommendation", "close_past", "close_future", "Price_Change", "Correct"]]

# Example usage: Backtest for '2024-01-01' with a 30-day future window
backtest_results = backtest_recommendations('2025-03-03', 3)
