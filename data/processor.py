import pandas as pd
import numpy as np
from typing import Tuple

def calculate_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Volume Weighted Average Price (VWAP) with daily reset
    
    Args:
        df: DataFrame with columns ['high', 'low', 'close', 'volume']
        Must be sorted by time ascending
        
    Returns:
        DataFrame with added 'vwap' column
    """
    df = df.sort_index()
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    
    # Calculate cumulative typical price * volume per day
    df['cumulative_vp'] = (df['typical_price'] * df['volume']).groupby(pd.Grouper(freq='D')).cumsum()
    df['cumulative_vol'] = df.groupby(pd.Grouper(freq='D'))['volume'].cumsum()
    
    # Debug print sample data
    sample = df.iloc[-5:] if len(df) > 5 else df
    print("\nVWAP Calculation Debug:")
    print(sample[['high', 'low', 'close', 'volume', 'typical_price', 'cumulative_vp', 'cumulative_vol']])
    
    # Handle days with no volume
    valid_volume = df['cumulative_vol'] > 0
    df['vwap'] = np.where(
        valid_volume,
        df['cumulative_vp'] / df['cumulative_vol'],
        df['typical_price']  # Fallback to typical price if no volume
    )
    
    print("\nFinal VWAP Sample:")
    print(df[['close', 'vwap']].tail())
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
