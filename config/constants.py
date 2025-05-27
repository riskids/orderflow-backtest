from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
COINAPI_KEY = os.getenv('COINAPI_KEY')
SYMBOL = os.getenv('SYMBOL', 'BYBIT_PERP_BTC_USDT')
TIMEFRAME = os.getenv('TIMEFRAME', '5MIN')

# Date Range
START_DATE = datetime.strptime(os.getenv('START_DATE'), '%Y-%m-%d')
END_DATE = datetime.strptime(os.getenv('END_DATE'), '%Y-%m-%d')

# Trading Sessions (UTC times)
ASIA_SESSION = tuple(map(int, os.getenv('ASIA_SESSION').split('-')))
NY_SESSION = tuple(map(int, os.getenv('NY_SESSION').split('-')))
