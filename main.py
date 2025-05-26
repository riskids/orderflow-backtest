from data.fetcher import fetch_ohlcv_data, fetch_order_book_data, merge_market_data
from data.processor import calculate_technical_indicators
from data.cache import check_cache, save_to_cache, load_from_cache, get_date_range_str
from strategies.range_poi import RangePOIStrategy
from backtest.engine import BacktestEngine
from typing import Tuple
import pandas as pd
from config.constants import START_DATE, END_DATE

def get_strategy(choice: int):
    """Get strategy instance based on user choice"""
    strategies = {
        1: RangePOIStrategy(),
    }
    return strategies.get(choice)

def run_strategy(df: pd.DataFrame, strategy) -> pd.DataFrame:
    """Run selected strategy on the data"""
    df['signal'] = 0
    
    for i in range(1, len(df)):
        current_hour = df.index[i].hour
        if strategy.is_in_session(current_hour):
            df.loc[df.index[i], 'signal'] = strategy.generate_signal(df, i)
    
    return df

def get_market_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Get market data with caching support"""
    start_str, end_str = get_date_range_str(START_DATE, END_DATE)
    
    # Check cache for OHLCV data
    if check_cache('ohlcv', start_str, end_str):
        print("Loading OHLCV data from cache...")
        ohlcv_df = load_from_cache('ohlcv', start_str, end_str)
    else:
        print("Fetching OHLCV data from API...")
        ohlcv_df = fetch_ohlcv_data()
        if ohlcv_df is not None:
            save_to_cache(ohlcv_df, 'ohlcv', start_str, end_str)
    
    # Check cache for order book data
    if check_cache('orderbook', start_str, end_str):
        print("Loading order book data from cache...")
        order_book_df = load_from_cache('orderbook', start_str, end_str)
    else:
        print("Fetching order book data from API...")
        order_book_df = fetch_order_book_data()
        if order_book_df is not None:
            save_to_cache(order_book_df, 'orderbook', start_str, end_str)
    
    return ohlcv_df, order_book_df

def main():
    try:
        print("Getting market data...")
        ohlcv_df, order_book_df = get_market_data()
        
        if ohlcv_df is not None:
            print("\nMerging and processing data...")
            df = merge_market_data(ohlcv_df, order_book_df)
            df = calculate_technical_indicators(df)
            
            print("\nAvailable strategies:")
            print("1. Range POI")
            choice = int(input("Select strategy (1-3): "))
            
            strategy = get_strategy(choice)
            print(f"\nRunning {strategy.strategy_name}...")
            df = run_strategy(df, strategy)
            
            print("\nRunning backtest...")
            engine = BacktestEngine()
            results = engine.run_backtest(df)
            engine.plot_results()
            
        else:
            print("Failed to fetch data. Please check your API key and connection.")
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
