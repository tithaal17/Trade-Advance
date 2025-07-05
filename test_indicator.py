import sqlite3
import numpy as np
import pandas as pd
import tulipy as ti
import config

def get_past_recommendations(start_date, end_date):
    """
    Fetch stock prices for a specific start_date, generate recommendations, and compare with end_date price.
    :param start_date: The date for which recommendations will be generated.
    :param end_date: The date to compare recommendations with.
    :return: List of recommendations.
    """

    conn = sqlite3.connect(config.DB_PATH)

    # Fetch stock prices from the start_date
    query = f"""
        SELECT s.symbol, sp.stock_id, sp.close
        FROM stock_price sp
        JOIN stock s ON s.id = sp.stock_id
        WHERE sp.date = '{start_date}';
    """
    df = pd.read_sql_query(query, conn)

    if df.empty:
        print(f"No stock data found for {start_date}.")
        conn.close()
        return []

    recommendations = []
    
    for _, row in df.iterrows():
        stock_id = row["stock_id"]
        symbol = row["symbol"]

        # Fetch historical close prices up to that date
        historical_query = f"""
            SELECT close FROM stock_price 
            WHERE stock_id = {stock_id} AND date <= '{start_date}'
            ORDER BY date ASC
        """
        prices_df = pd.read_sql_query(historical_query, conn)

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
            if macd[-1] > signal[-1] and macd[-2] <= signal[-2]:
                recommendation = "Buy"
                indicator_used = "MACD"
            elif macd[-1] < signal[-1] and macd[-2] >= signal[-2]:
                recommendation = "Sell"
                indicator_used = "MACD"

        recommendations.append({
            "symbol": symbol,
            "stock_id": stock_id,
            "past_price": row["close"],
            "recommendation": recommendation,
            "indicator_used": indicator_used
        })

    conn.close()
    return recommendations

def compare_with_fixed_date(recommendations, end_date):
    """
    Compare past recommendations with a fixed end_date price.
    :param recommendations: List of recommendations from get_past_recommendations().
    :param end_date: The date to compare recommendations against.
    :return: Accuracy statistics.
    """

    conn = sqlite3.connect(config.DB_PATH)
    correct_count = 0
    total_count = len(recommendations)

    for rec in recommendations:
        stock_id = rec["stock_id"]
        past_price = rec["past_price"]
        recommendation = rec["recommendation"]

        # Fetch closing price on the end_date
        query = f"""
            SELECT close FROM stock_price 
            WHERE stock_id = {stock_id} AND date = '{end_date}';
        """
        cursor = conn.cursor()
        cursor.execute(query)
        result = cursor.fetchone()

        if result:
            current_price = result[0]
            price_change = (current_price - past_price) / past_price  # % change

            # Check if the recommendation was correct
            success_threshold = 0.01  # 1% movement
            if (recommendation == "Buy" and price_change > success_threshold) or \
               (recommendation == "Sell" and price_change < -success_threshold):
                correct_count += 1

    conn.close()

    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
    return {
        "total_recommendations": total_count,
        "correct_recommendations": correct_count,
        "accuracy": accuracy
    }

# Example usage:
start_date = "2024-04-09"  # Replace with desired past date
end_date = "2025-03-07"  # Replace with desired comparison date

past_recommendations = get_past_recommendations(start_date, end_date)
accuracy_result = compare_with_fixed_date(past_recommendations, end_date)

print(f"Accuracy from {start_date} to {end_date}: {accuracy_result}")
