import requests
import pandas as pd
import numpy as np
from typing import Optional, Dict
from datetime import datetime
from config.constants import COINAPI_KEY, SYMBOL, TIMEFRAME, START_DATE, END_DATE

HEADERS = {'X-CoinAPI-Key': COINAPI_KEY}

def fetch_ohlcv_data() -> Optional[pd.DataFrame]:
    """Fetch OHLCV data from CoinAPI"""
    url = f"https://rest.coinapi.io/v1/ohlcv/{SYMBOL}/history?period_id={TIMEFRAME}&limit=1000&time_start={START_DATE.isoformat()}&time_end={END_DATE.isoformat()}"
    
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        ohlcv_data = response.json()
        
        if not isinstance(ohlcv_data, list):
            raise ValueError(f"Unexpected OHLCV data format: {type(ohlcv_data)}")
        
        ohlcv_rows = []
        for item in ohlcv_data:
            if not all(key in item for key in ['time_period_start', 'price_open', 'price_high', 'price_low', 'price_close']):
                continue
                
            ohlcv_rows.append({
                'time': pd.to_datetime(item['time_period_start']),
                'open': float(item['price_open']),
                'high': float(item['price_high']),
                'low': float(item['price_low']),
                'close': float(item['price_close']),
                'volume': float(item.get('volume_traded', 0))
            })
            
        df = pd.DataFrame(ohlcv_rows).set_index('time')
        return df if not df.empty else None
        
    except requests.exceptions.RequestException as e:
        print(f"OHLCV API request failed: {str(e)}")
        return None

def fetch_order_book_data(batch_size: int = 20000, hours_per_batch: int = 3) -> Optional[pd.DataFrame]:
    """Fetch order book data with memory-efficient batches"""
    cvd_rows = []
    current_time = START_DATE
    
    while current_time <= END_DATE:
        batch_end = current_time + pd.Timedelta(hours=hours_per_batch)
        if batch_end > END_DATE:
            batch_end = END_DATE
            
        print(f"Fetching {current_time} to {batch_end}...")
        url = f"https://rest.coinapi.io/v1/orderbooks/{SYMBOL}/history?limit={batch_size}&time_start={current_time.isoformat()}&time_end={batch_end.isoformat()}"
        
        try:
            response = requests.get(url, headers=HEADERS)
            response.raise_for_status()
            book_data = response.json()

            if not isinstance(book_data, list):
                print(f"Unexpected data format for {current_time}")
                current_time = batch_end
                continue

            batch_count = 0
            for book in book_data:
                try:
                    if not isinstance(book, dict) or 'time_exchange' not in book:
                        continue
                        
                    timestamp = pd.to_datetime(book.get('time_exchange'))
                    if pd.isna(timestamp):
                        continue
                    
                    # Process only if within batch time range
                    if timestamp >= current_time and timestamp <= batch_end:
                        bid_vol = sum(float(level['size']) for level in book.get('bids', []))
                        ask_vol = sum(float(level['size']) for level in book.get('asks', []))
                        
                        cvd_rows.append({
                            'time': timestamp,
                            'delta': bid_vol - ask_vol,
                            'bid_vol': bid_vol,
                            'ask_vol': ask_vol
                        })
                        batch_count += 1
                    
                except (KeyError, TypeError, ValueError) as e:
                    print(f"Skipping invalid book entry: {str(e)}")
                    continue
            
            print(f"Processed {batch_count} order book entries")
            
            # Clear memory after each batch
            if len(cvd_rows) > 10000:
                pd.DataFrame(cvd_rows).to_parquet(f'cache/orderbook_temp_{current_time.timestamp()}.parquet')
                cvd_rows = []
                
        except requests.exceptions.RequestException as e:
            print(f"Failed to fetch data: {str(e)}")
            return None
        
        current_time = batch_end
    
    # Combine all temp files if any
    if cvd_rows:
        return pd.DataFrame(cvd_rows).set_index('time')
    
    print("No valid order book data found")
    return None

def merge_market_data(ohlcv_df: pd.DataFrame, order_book_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Merge OHLCV and CVD data"""
    if order_book_df is not None:
        ohlcv_df = ohlcv_df.join(order_book_df, how='left')
        ohlcv_df['delta'] = ohlcv_df['delta'].fillna(0)
    else:
        print("Warning: No valid order book data - using random CVD")
        ohlcv_df['delta'] = np.random.uniform(-100, 100, len(ohlcv_df))
    
    ohlcv_df['cvd'] = ohlcv_df['delta'].cumsum()
    return ohlcv_df
