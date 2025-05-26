from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional
from config.constants import ASIA_SESSION, NY_SESSION

class BaseStrategy(ABC):
    """Base class for trading strategies"""
    
    def __init__(self):
        self.strategy_name = "Base Strategy"
        self.session_ranges = {
            'asia': ASIA_SESSION,
            'ny': NY_SESSION
        }
    
    def is_in_session(self, hour: int) -> bool:
        """Check if current hour is in trading session"""
        in_asia = self.session_ranges['asia'][0] <= hour < self.session_ranges['asia'][1]
        in_ny = self.session_ranges['ny'][0] <= hour < self.session_ranges['ny'][1]
        return in_asia or in_ny
    
    @abstractmethod
    def generate_signal(self, df: pd.DataFrame, i: int) -> int:
        """Generate trading signal for given index"""
        pass
    
    def get_required_cols(self) -> list:
        """Return list of required columns for the strategy"""
        return []
