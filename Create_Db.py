import sqlite3
import config

connection = sqlite3.connect(config.DB_PATH)

cursor = connection.cursor()

cursor.execute("""
               CREATE TABLE IF NOT EXISTS stock(
                    id INTEGER PRIMARY KEY,
                    symbol TEXT NOT NULL UNIQUE,
                    name TEXT NOT NULL,
                    exchange TEXT NOT NULL,
                    shortable BOOLEAN NOT NULL
                )
            """)

cursor.execute("""
               CREATE TABLE IF NOT EXISTS stock_price (
                id INTEGER PRIMARY KEY,
                stock_id INTEGER,
                date TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume INTEGER NOT NULL,
                sma_20 REAL,
                sma_50 REAL,
                rsi_14 REAL,
                FOREIGN KEY(stock_id) REFERENCES stock(id),
                UNIQUE(stock_id, date)  
            )
    """)


cursor.execute("""
    CREATE TABLE IF NOT EXISTS user (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        cash REAL NOT NULL DEFAULT 10000.00
    );
""")
cursor.execute("""
    CREATE TABLE IF NOT EXISTS stock_holding (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        company_symbol TEXT NOT NULL,
        number_of_shares INTEGER DEFAULT 0,
        investment_amount REAL DEFAULT 0,
        buying_value TEXT DEFAULT '[]',  -- JSON as TEXT for SQLite
        FOREIGN KEY (user_id) REFERENCES user(id) ON DELETE CASCADE
    );

""")
cursor.execute("""
    CREATE TABLE IF NOT EXISTS transactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        symbol TEXT NOT NULL,
        shares INTEGER NOT NULL,
        price REAL NOT NULL,
        transaction_type TEXT CHECK (transaction_type IN ('BUY', 'SELL', 'SHORT', 'COVER')) NOT NULL,
        profit_loss REAL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES user(id) ON DELETE CASCADE
    );
""")

cursor.execute("""
    CREATE TABLE IF NOT EXISTS strategy(
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL               
    )
""")

cursor.execute("""
    CREATE TABLE IF NOT EXISTS stock_strategy (
        stock_id INTEGER NOT NULL,
        strategy_id INTEGER NOT NULL,
        FOREIGN KEY (stock_id) REFERENCES stock(id),
        FOREIGN KEY (strategy_id) REFERENCES strategy(id)                
    )
""")

    # Create table if it doesn't exist
cursor.execute("""
    CREATE TABLE IF NOT EXISTS trade_signals (
        id INTEGER PRIMARY KEY,
        stock_id INTEGER,
        signal TEXT NOT NULL,
        indicator TEXT NOT NULL,
        timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(stock_id) REFERENCES stock(id)
    )
""")

connection.commit()