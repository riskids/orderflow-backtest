import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict

class BacktestEngine:
    """Backtesting engine for trading strategies"""
    
    def __init__(self):
        self.results = None
        
    def run_backtest(self, df: pd.DataFrame) -> Dict:
        """
        Run backtest on strategy signals
        
        Args:
            df: DataFrame with 'signal' column
            
        Returns:
            Dictionary of backtest results
        """
        # Calculate positions and returns
        df['position'] = df['signal'].shift(1)
        df['returns'] = df['close'].pct_change() * df['position']
        df['equity'] = (1 + df['returns']).cumprod()
        
        # Calculate performance metrics
        stats = self._calculate_stats(df)
        
        # Store results
        self.results = {
            'df': df,
            'stats': stats
        }
        
        return self.results
    
    def _calculate_stats(self, df: pd.DataFrame) -> Dict:
        """Calculate performance statistics"""
        trades = df[df['position'] != 0]
        winning_trades = trades[trades['returns'] > 0]
        losing_trades = trades[trades['returns'] < 0]
        
        total_trades = len(trades)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        avg_win = winning_trades['returns'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['returns'].mean() if len(losing_trades) > 0 else 0
        profit_factor = -avg_win / avg_loss if avg_loss != 0 else np.inf
        
        max_dd = (df['equity'].cummax() - df['equity']).max()
        sharpe = np.sqrt(252) * df['returns'].mean() / df['returns'].std() if df['returns'].std() != 0 else 0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_dd,
            'sharpe_ratio': sharpe,
            'total_return': df['equity'].iloc[-1] - 1
        }
    
    def plot_results(self):
        """Plot backtest results"""
        if self.results is None:
            raise ValueError("No backtest results available. Run backtest first.")
            
        df = self.results['df']
        stats = self.results['stats']
        
        plt.figure(figsize=(14, 7))
        plt.plot(df['equity'], label='Equity Curve', color='blue')
        
        # Plot trades
        winning_trades = df[df['returns'] > 0]
        losing_trades = df[df['returns'] < 0]
        
        plt.scatter(winning_trades.index, winning_trades['equity'], 
                   label='Winning Trades', marker='^', color='green')
        plt.scatter(losing_trades.index, losing_trades['equity'], 
                   label='Losing Trades', marker='v', color='red')
        
        plt.title('Strategy Backtest Results')
        plt.ylabel('Equity Growth')
        plt.legend()
        plt.grid()
        plt.show()
        
        # Print statistics
        print("\n=== Backtest Results ===")
        for k, v in stats.items():
            if isinstance(v, float):
                print(f"{k.replace('_', ' ').title()}: {v*100:.2f}%" if k != 'sharpe_ratio' else f"Sharpe Ratio: {v:.2f}")
            else:
                print(f"{k.replace('_', ' ').title()}: {v}")
