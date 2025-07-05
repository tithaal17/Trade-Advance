from fastapi import APIRouter, Request, Query, HTTPException
from fastapi.responses import JSONResponse
import sqlite3, config
from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory="templates")

router = APIRouter()

def get_db_connection():
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@router.get("/index")
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
    
@router.get("/search_stocks")
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

@router.get("/stock/{symbol}")
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
