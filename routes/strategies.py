from fastapi import APIRouter, Request, Query, HTTPException
from fastapi import  Request, Form, Depends
from fastapi.responses import RedirectResponse
import sqlite3
from fastapi.templating import Jinja2Templates
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus
import sqlite3, config

templates = Jinja2Templates(directory="templates")

router = APIRouter()

def get_db_connection():
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@router.post("/apply_strategy")
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

@router.get("/strategies")
def strategies(request:Request):

    connection = get_db_connection()
    cursor = connection.cursor()

    cursor.execute("""
        select * from strategy
    """)

    strategies = cursor.fetchall()

    return templates.TemplateResponse("strategies.html",{"request":request, "strategies" : strategies})  

@router.get("/orders")
def orders(request:Request):

    trading_client = TradingClient(config.API_KEY, config.SECRET_KEY, paper=True)

    orders_request = GetOrdersRequest(status=QueryOrderStatus.ALL)
    orders = trading_client.get_orders(filter=orders_request)

    return templates.TemplateResponse("orders.html",{"request":request, "orders" : orders})  
    
@router.get("/strategy/{strategy_id}")
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
