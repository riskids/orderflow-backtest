from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
COINAPI_KEY = os.getenv('COINAPI_KEY')
SYMBOL = os.getenv('SYMBOL', 'BYBIT_PERP_BTC_USDT')
TIMEFRAME = os.getenv('TIMEFRAME', '5MIN')

# Date Range
START_DATE = datetime(2025, 2, 27)
END_DATE = datetime(2025, 4, 20)

# Trading Sessions (UTC times)
ASIA_SESSION = (0, 8)    # 00:00-08:00 UTC
NY_SESSION = (12, 20)    # 12:00-20:00 UTC
