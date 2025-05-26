import pandas as pd
import numpy as np
from typing import Tuple
from strategies.base_strategy import BaseStrategy
from data.processor import calculate_vwap, calculate_volume_profile

class RangePOIStrategy(BaseStrategy):
    """Range Trading Strategy using Volume Profile POIs and Order Flow"""
    
    def __init__(self):
        super().__init__()
        self.strategy_name = "Range POI Strategy"
        self.risk_per_trade = 0.01  # 1% risk
        self.min_rr = 2.5
        
    def get_required_cols(self) -> list:
        """Return list of required columns"""
        return ['open', 'high', 'low', 'close', 'volume', 'delta', 'vwap']
        
    def detect_swing_points(self, df: pd.DataFrame, lookback: int = 5) -> Tuple[float, float]:
        """
        Identify recent swing highs and lows
        """
        df['swing_high'] = df['high'].rolling(lookback, center=True).max()
        df['swing_low'] = df['low'].rolling(lookback, center=True).min()
        return df['swing_high'].iloc[-1], df['swing_low'].iloc[-1]
        
    def get_monday_levels(self, df: pd.DataFrame) -> Tuple[float, float]:
        """
        Get Monday's high and low from weekly data
        """
        mondays = df[df.index.dayofweek == 0]
        if not mondays.empty:
            return mondays['high'].max(), mondays['low'].min()
        return np.nan, np.nan
        
    def calculate_vwap_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate VWAP ±1 levels
        """
        df['vwap_std'] = df['close'].rolling(20).std()
        df['vwap_upper'] = df['vwap'] + df['vwap_std']
        df['vwap_lower'] = df['vwap'] - df['vwap_std']
        return df
        
    def detect_trapped_delta(self, df: pd.DataFrame, i: int) -> bool:
        """
        Detect trapped delta (absorption) patterns
        """
        if i < 2:
            return False
            
        # Bullish absorption: price down but delta increasing
        bull_trap = (df['close'].iloc[i] < df['close'].iloc[i-1]) and \
                   (df['delta'].iloc[i] > df['delta'].iloc[i-1])
                   
        # Bearish absorption: price up but delta decreasing
        bear_trap = (df['close'].iloc[i] > df['close'].iloc[i-1]) and \
                   (df['delta'].iloc[i] < df['delta'].iloc[i-1])
                   
        return bull_trap or bear_trap
        
    def generate_signal(self, df: pd.DataFrame, i: int) -> int:
        """
        Generate trading signal based on POI and order flow
        """
        if i < 20:  # Need enough data
            return 0
            
        # Calculate all POIs
        df = self.calculate_vwap_bands(df)
        val, vah, poc = calculate_volume_profile(df)
        swing_high, swing_low = self.detect_swing_points(df)
        monday_high, monday_low = self.get_monday_levels(df)
        
        # Current price action
        current_close = df['close'].iloc[i]
        current_delta = df['delta'].iloc[i]
        prev_delta = df['delta'].iloc[i-1]
        
        # Check for POI touches with confluence
        for level in [vah, val, swing_high, swing_low, monday_high, monday_low,
                     df['vwap_upper'].iloc[i], df['vwap'].iloc[i], df['vwap_lower'].iloc[i]]:
            
            if np.isnan(level):
                continue
                
            # Buy signal: at support with bullish confluence
            if (abs(current_close - level) < 0.0015 * current_close and  # Within 0.15% of level
                current_delta > prev_delta and  # Delta increasing
                self.detect_trapped_delta(df, i)):  # Absorption
                
                return 1  # Buy
                
            # Sell signal: at resistance with bearish confluence
            elif (abs(current_close - level) < 0.0015 * current_close and
                  current_delta < prev_delta and  # Delta decreasing
                  self.detect_trapped_delta(df, i)):  # Absorption
                  
                return -1  # Sell
                
        return 0  # No signal
