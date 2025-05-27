import requests
import pandas as pd
import numpy as np
from typing import Optional, Dict
from datetime import datetime
from config.constants import COINAPI_KEY, SYMBOL, TIMEFRAME, START_DATE, END_DATE
from tqdm import tqdm

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

class OrderBookFetcher:
    def __init__(self):
        pass
        
    def fetch_order_book_data_at_time(self, timestamp: pd.Timestamp, window: int = 300, batch_size: int = 10000) -> Optional[Dict]:
        """
        Fetch order book data at specific timestamp with surrounding window (in seconds)
        Returns single data point with delta, bid_vol, ask_vol
        Processes data in 30-minute batches for memory efficiency
        """
        start_time = timestamp - pd.Timedelta(seconds=window)
        end_time = timestamp
        
        cvd_rows = []
        current_time = start_time
        total_batches = ((end_time - start_time).total_seconds() // (30 * 60)) + 1
        batch_num = 1
        
        while current_time < end_time:
            batch_end = current_time + pd.Timedelta(minutes=5)
            if batch_end > end_time:
                batch_end = end_time
                
            print(f"\nProcessing batch {batch_num}/{int(total_batches)}")
            print(f"Time range: {current_time} to {batch_end}")
                
            date_str = current_time.strftime('%Y-%m-%dT00:00:00.0000000Z')
            url = f"https://rest.coinapi.io/v1/orderbooks/{SYMBOL}/history?date={date_str}&limit={batch_size}&time_start={current_time.strftime('%Y-%m-%dT%H:%M:%S.0000000Z')}&time_end={batch_end.strftime('%Y-%m-%dT%H:%M:%S.0000000Z')}&limit_levels=1"
            
            try:
                response = requests.get(url, headers=HEADERS)
                response.raise_for_status()
                book_data = response.json()

                if not isinstance(book_data, list):
                    print(f"Unexpected data format for batch {current_time}-{batch_end}")
                    current_time = batch_end
                    continue

                print(f"\nFirst book entry sample: {book_data[0] if book_data else 'Empty'}")  # Debug first entry
                
                for book in book_data:
                    try:
                        if not isinstance(book, dict):
                            print("Skipping non-dict book entry")
                            continue
                            
                        if 'time_exchange' not in book:
                            print("Skipping book entry without time_exchange")
                            continue
                            
                        book_time = pd.to_datetime(book.get('time_exchange'))
                        if pd.isna(book_time):
                            print("Skipping book entry with invalid time")
                            continue
                        
                        bids = book.get('bids', [])
                        asks = book.get('asks', [])
                        
                        if not bids and not asks:
                            print(f"Skipping empty book at {book_time}")
                            continue
                            
                        # Calculate volumes with type checking
                        bid_vol = 0.0
                        ask_vol = 0.0
                        
                        for level in bids:
                            try:
                                bid_vol += float(level.get('size', 0))
                            except (TypeError, ValueError) as e:
                                print(f"Invalid bid size: {level.get('size')} - {str(e)}")
                                
                        for level in asks:
                            try:
                                ask_vol += float(level.get('size', 0))
                            except (TypeError, ValueError) as e:
                                print(f"Invalid ask size: {level.get('size')} - {str(e)}")
                        
                        if bid_vol == 0 and ask_vol == 0:
                            print(f"Skipping zero-volume book at {book_time}")
                            continue
                            
                        # Collect all data during batch processing
                        cvd_rows.append({
                            'time': book_time,
                            'delta': bid_vol - ask_vol,
                            'bid_vol': bid_vol,
                            'ask_vol': ask_vol
                        })
                            
                    except Exception as e:
                        print(f"Error processing book entry: {str(e)}")
                        print(f"Problematic book entry: {book}")
                        continue
                        
                # Print batch CVD data
                print("\nBatch CVD Summary:")
                print(pd.DataFrame(cvd_rows[-10:]).to_string())  # Print last 10 entries
                # print(book_data)  # Print last 10 entries
                
                # Clear memory after each batch
                if len(cvd_rows) > 10000:
                    pd.DataFrame(cvd_rows).to_parquet(f'cache/orderbook_temp_{current_time.timestamp()}.parquet')
                    cvd_rows = []
                    
            except requests.exceptions.RequestException as e:
                print(f"Failed to fetch batch {current_time}-{batch_end}: {str(e)}")
                return None
                
            current_time = batch_end
        
        if not cvd_rows:
            return None
            
        # Combine all data and calculate totals
        df = pd.DataFrame(cvd_rows)
        matched = df[abs(df['time'] - timestamp) <= pd.Timedelta(seconds=window)]
        
        if matched.empty:
            print("mathced empty")
            return None
            
        return {
            'time': timestamp,
            'delta': matched['delta'].sum(),
            'bid_vol': matched['bid_vol'].sum(),
            'ask_vol': matched['ask_vol'].sum()
        }


def fetch_order_book_data(batch_size: int = 10000, hours_per_batch: int = 1) -> Optional[pd.DataFrame]:
    """Fetch order book data with memory-efficient batches"""
    cvd_rows = []
    current_time = START_DATE
    
    while current_time <= END_DATE:
        batch_end = current_time + pd.Timedelta(hours=hours_per_batch)
        if batch_end > END_DATE:
            batch_end = END_DATE
            
        print(f"Fetching {current_time} to {batch_end}...")
        date_str = current_time.strftime('%Y-%m-%dT00:00:00.0000000Z')
        url = f"https://rest.coinapi.io/v1/orderbooks/{SYMBOL}/history?date={date_str}&limit={batch_size}&time_start={current_time.strftime('%Y-%m-%dT%H:%M:%S.0000000Z')}&time_end={batch_end.strftime('%Y-%m-%dT%H:%M:%S.0000000Z')}"
        
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
    # Initialize delta column with zeros - will be updated by strategy
    ohlcv_df['delta'] = 0
    ohlcv_df['cvd'] = 0
    return ohlcv_df
