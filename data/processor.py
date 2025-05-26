import pandas as pd
import numpy as np
from typing import Tuple

def calculate_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Volume Weighted Average Price (VWAP) with daily reset
    
    Args:
        df: DataFrame with columns ['high', 'low', 'close', 'volume']
        
    Returns:
        DataFrame with added 'vwap' column
    """
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['cumulative_vp'] = df.groupby(pd.Grouper(freq='D'))['typical_price'].cumsum()
    df['cumulative_vol'] = df.groupby(pd.Grouper(freq='D'))['volume'].cumsum()
    df['vwap'] = df['cumulative_vp'] / df['cumulative_vol']
    return df

def calculate_volume_profile(df: pd.DataFrame, bins: int = 20) -> Tuple[float, float, float]:
    """
    Calculate Volume Profile (VAL, VAH, POC)
    
    Args:
        df: DataFrame with columns ['high', 'low', 'close', 'volume']
        bins: Number of bins for volume distribution
        
    Returns:
        Tuple of (value_area_low, value_area_high, point_of_control)
    """
    vp_range = np.linspace(df['low'].min(), df['high'].max(), bins)
    volume_dist = pd.cut(df['close'], bins=vp_range, labels=vp_range[:-1])
    df['vp_bin'] = volume_dist
    volume_per_bin = df.groupby('vp_bin', observed=False)['volume'].sum()
    
    poc_bin = volume_per_bin.idxmax()
    return (vp_range[0], vp_range[-1], poc_bin)

def calculate_volume_ma(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Calculate Volume Moving Average
    
    Args:
        df: DataFrame with 'volume' column
        window: MA window size
        
    Returns:
        DataFrame with added 'volume_ma' column
    """
    df['volume_ma'] = df['volume'].rolling(window).mean()
    return df

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all technical indicators and add to DataFrame
    
    Args:
        df: Input DataFrame with market data
        
    Returns:
        DataFrame with added technical indicators
    """
    df = calculate_vwap(df)
    _, _, poc = calculate_volume_profile(df)
    df['poc'] = poc
    df = calculate_volume_ma(df)
    return df
