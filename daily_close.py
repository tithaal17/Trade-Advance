 
import config
from alpaca.trading.client import TradingClient

trading_client = TradingClient(config.API_KEY, config.SECRET_KEY, paper=True)

positions = trading_client.get_all_positions()
print(positions)

response = trading_client.close_all_positions()

print(response)
