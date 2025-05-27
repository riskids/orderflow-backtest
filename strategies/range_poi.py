import pandas as pd
import numpy as np
from typing import Tuple
from strategies.base_strategy import BaseStrategy
from data.processor import calculate_vwap, calculate_volume_profile

class RangePOIStrategy(BaseStrategy):
    """Range Trading Strategy using Volume Profile POIs and Order Flow"""
    
    def __init__(self, fetcher=None):
        super().__init__()
        self.strategy_name = "Range POI Strategy"
        self.risk_per_trade = 0.01  # 1% risk
        self.min_rr = 2
        self.fetcher = fetcher
        
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
        Calculate VWAP ±1 levels using daily anchor VWAP (TradingView style)
        """
        # Calculate daily VWAP and standard deviation bands like TradingView
        def calculate_daily_vwap_bands(group):
            # Calculate typical price
            typical_price = (group['high'] + group['low'] + group['close']) / 3
            volume = group['volume']
            
            # Calculate cumulative VWAP
            cum_vol = volume.cumsum()
            cum_tp_vol = (typical_price * volume).cumsum()
            vwap = cum_tp_vol / cum_vol
            
            # Calculate standard deviation from VWAP (TradingView method)
            # Sum of squared deviations weighted by volume
            squared_deviations = (typical_price - vwap) ** 2
            cum_sq_dev_vol = (squared_deviations * volume).cumsum()
            variance = cum_sq_dev_vol / cum_vol
            std_dev = np.sqrt(variance)
            
            # Standard deviation bands with multiplier = 1
            multiplier = 1.0
            upper_band = vwap + (multiplier * std_dev)
            lower_band = vwap - (multiplier * std_dev)
            
            return pd.DataFrame({
                'daily_vwap': vwap,
                'vwap_std': std_dev,
                'vwap_upper': upper_band,
                'vwap_lower': lower_band
            }, index=group.index)
        
        # Group by day and calculate VWAP bands
        daily_bands = df.groupby(pd.Grouper(freq='D')).apply(calculate_daily_vwap_bands)
        daily_bands.index = daily_bands.index.droplevel(0)
        
        # Merge back to original dataframe
        df = df.join(daily_bands)
        
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
        
        # Define POI levels with their names
        poi_levels = [
            ('VAH', vah),
            ('VAL', val),
            ('Swing High', swing_high),
            ('Swing Low', swing_low),
            ('Monday High', monday_high),
            ('Monday Low', monday_low),
            ('VWAP Upper', df['vwap_upper'].iloc[i]),
            ('VWAP', df['vwap'].iloc[i]),
            ('VWAP Lower', df['vwap_lower'].iloc[i])
        ]
        
        # Check for POI touches with confluence
        for poi_name, level in poi_levels:
            if np.isnan(level):
                continue
                
            # Check if price is near POI level
            threshold = 0.0055 * current_close  # Increased from 0.15% to 1% for testing
            diff = abs(current_close - level)
            print(f"Checking {poi_name}: price={current_close:.2f}, level={level:.2f}, diff={diff:.4f}, threshold={threshold:.4f}, at {pd.to_datetime(df.index[i])}")
            
            if diff < threshold:
                print(f"POI HIT! {poi_name} at {level:.2f} (diff: {diff:.4f})")
                # Fetch real-time order book data near POI
                if self.fetcher and hasattr(df.index, 'to_pydatetime'):
                    try:
                        # Ensure timestamp is in correct format
                        timestamp = pd.to_datetime(df.index[i])
                        ob_data = self.fetcher.fetch_order_book_data_at_time(timestamp)
                        print(f"Fetcher returned: {ob_data}")
                        print(f"Fetching OB for timestamp: {timestamp}, type: {type(timestamp)}")

                        
                        if ob_data and 'delta' in ob_data:
                            current_delta = ob_data['delta']
                            prev_delta = df['delta'].iloc[i-1] if i > 0 else 0
                        else:
                            print(f"No order book data for {timestamp}")
                            continue
                    except Exception as e:
                        print(f"Failed to fetch order book at {df.index[i]}: {str(e)}")
                        continue
                
                # Buy signal: at support with bullish confluence
                if (current_delta > prev_delta and  # Delta increasing
                    self.detect_trapped_delta(df, i)):  # Absorption
                    print(f"BUY! delta trapped at price={current_close:.2f}")

                    return 1  # Buy
                    
                # Sell signal: at resistance with bearish confluence
                elif (current_delta < prev_delta and  # Delta decreasing
                      self.detect_trapped_delta(df, i)):  # Absorption
                    print(f"SELL! delta trapped at price={current_close:.2f}")
                    return -1  # Sell
                
        return 0  # No signal
