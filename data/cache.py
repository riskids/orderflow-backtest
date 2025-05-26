import os
import pandas as pd
from datetime import datetime
from typing import Optional, Tuple
import json

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_cache_filename(data_type: str, start_date: str, end_date: str) -> str:
    """Generate cache filename from date range"""
    return f"{CACHE_DIR}/{data_type}_{start_date}_{end_date}.parquet"

def check_cache(data_type: str, start_date: str, end_date: str) -> bool:
    """Check if cached data exists for date range"""
    cache_file = get_cache_filename(data_type, start_date, end_date)
    return os.path.exists(cache_file)

def save_to_cache(data: pd.DataFrame, data_type: str, start_date: str, end_date: str):
    """Save data to cache"""
    cache_file = get_cache_filename(data_type, start_date, end_date)
    data.to_parquet(cache_file)

def load_from_cache(data_type: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Load data from cache"""
    cache_file = get_cache_filename(data_type, start_date, end_date)
    return pd.read_parquet(cache_file)

def get_date_range_str(start_date: datetime, end_date: datetime) -> Tuple[str, str]:
    """Convert datetime objects to string format for filenames"""
    return (start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d"))
