#!/usr/bin/env python3
"""
MicroEdge - Momentum Trading Strategy

Strategy Type: momentum
Description: TSLA Momentum Pulse. A mid-frequency, symmetric momentum strategy designed for TSLA. The strategy enters long after 3 consecutive up-closes, anticipating short-term trend continuation. Conversely, it enters short after 3 consecutive down-closes, positioning for downside extension.

Each trade is exited after 2 days, or earlier if the price reverses direction (i.e., closes against the position). The approach is deliberately minimalistic and robust, capturing recurring behavioral flows in a volatile large-cap stock without reliance on indicators or overlays.
Created: 2025-07-03T11:21:21.026Z

WARNING: This is a template implementation. Thoroughly backtest before live trading.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('strategy.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MicroEdgeStrategy:
    """
    MicroEdge Implementation
    
    Strategy Type: momentum
    Risk Level: Monitor drawdowns and position sizes carefully
    """
    
    def __init__(self, config=None):
        self.config = config or self.get_default_config()
        self.positions = {}
        self.performance_metrics = {}
        logger.info(f"Initialized MicroEdge strategy")
        
    def get_default_config(self):
        """Default configuration parameters"""
        return {
            'max_position_size': 0.05,  # 5% max position size
            'stop_loss_pct': 0.05,      # 5% stop loss
            'lookback_period': 20,       # 20-day lookback
            'rebalance_freq': 'daily',   # Rebalancing frequency
            'transaction_costs': 0.001,  # 0.1% transaction costs
        }
    
    def load_data(self, symbols, start_date, end_date):
        """Load market data for analysis"""
        try:
            import yfinance as yf
            data = yf.download(symbols, start=start_date, end=end_date)
            logger.info(f"Loaded data for {len(symbols)} symbols")
            return data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None

# =============================================================================
# USER'S STRATEGY IMPLEMENTATION
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# Strategy class definition
class MicroEdgeMomentumStrategy:
    def __init__(self, data, initial_capital=100000, position_size_pct=0.1, slippage=0.0, commission=0.0):
        # Initialize strategy parameters
        self.data = data.copy()
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.slippage = slippage
        self.commission = commission
        self.signals = pd.DataFrame(index=self.data.index)
        self.trades = []
        self.equity_curve = None
        self.positions = None

    def generate_signals(self):
        # Generate momentum signals: 3 consecutive up or down closes
        close = self.data['Close']
        up = close > close.shift(1)
        down = close < close.shift(1)
        up3 = up & up.shift(1) & up.shift(2)
        down3 = down & down.shift(1) & down.shift(2)
        self.signals['long_entry'] = up3.astype(int)
        self.signals['short_entry'] = down3.astype(int)
        self.signals['signal'] = 0
        self.signals.loc[self.signals['long_entry'] == 1, 'signal'] = 1
        self.signals.loc[self.signals['short_entry'] == 1, 'signal'] = -1
        self.signals['signal'] = self.signals['signal'].shift(1).fillna(0)
        # 1 = long entry, -1 = short entry, 0 = no entry

    def backtest(self):
        # Backtest the strategy
        capital = self.initial_capital
        position = 0
        entry_price = 0
        entry_index = None
        trade_log = []
        equity = []
        positions = []
        returns = []
        for i in range(len(self.data)):
            date = self.data.index[i]
            price = self.data['Close'].iloc[i]
            signal = self.signals['signal'].iloc[i]
            # Entry logic
            if position == 0:
                if signal == 1:
                    # Enter long
                    size = (capital * self.position_size_pct) // price
                    if size > 0:
                        entry_price = price * (1 + self.slippage)
                        position = size
                        entry_index = i
                        trade_log.append({'entry_date': date, 'entry_price': entry_price, 'position': position, 'side': 'long'})
                        logging.info("Long entry on %s at %.2f, size %d" % (date, entry_price, position))
                elif signal == -1:
                    # Enter short
                    size = (capital * self.position_size_pct) // price
                    if size > 0:
                        entry_price = price * (1 - self.slippage)
                        position = -size
                        entry_index = i
                        trade_log.append({'entry_date': date, 'entry_price': entry_price, 'position': position, 'side': 'short'})
                        logging.info("Short entry on %s at %.2f, size %d" % (date, entry_price, position))
            # Exit logic
            elif position > 0:
                # Long position
                exit = False
                # Exit after 2 days
                if i - entry_index >= 2:
                    exit = True
                # Or if price closes below previous close (reversal)
                elif price < self.data['Close'].iloc[i-1]:
                    exit = True
                if exit:
                    exit_price = price * (1 - self.slippage)
                    pnl = (exit_price - entry_price) * position - self.commission
                    capital += pnl
                    trade_log[-1].update({'exit_date': date, 'exit_price': exit_price, 'pnl': pnl})
                    logging.info("Long exit on %s at %.2f, PnL %.2" % (date, exit_price, pnl))
                    position = 0
                    entry_price = 0
                    entry_index = None
            elif position < 0:
                # Short position
                exit = False
                # Exit after 2 days
                if i - entry_index >= 2:
                    exit = True
                # Or if price closes above previous close (reversal)
                elif price > self.data['Close'].iloc[i-1]:
                    exit = True
                if exit:
                    exit_price = price * (1 + self.slippage)
                    pnl = (entry_price - exit_price) * abs(position) - self.commission
                    capital += pnl
                    trade_log[-1].update(" + str('exit_date': date, 'exit_price': exit_price, 'pnl': pnl) + ")
                    logging.info("Short exit on %s at %.2f, PnL %.2" % (date, exit_price, pnl))
                    position = 0
                    entry_price = 0
                    entry_index = None
            # Track equity and positions
            if position == 0:
                equity.append(capital)
                positions.append(0)
                returns.append(0)
            else:
                # Mark-to-market
                if position > 0:
                    mtm = (price - entry_price) * position
                else:
                    mtm = (entry_price - price) * abs(position)
                equity.append(capital + mtm)
                positions.append(position)
                returns.append((equity[-1] - equity[-2]) / equity[-2] if len(equity) > 1 else 0)
        self.trades = trade_log
        self.equity_curve = pd.Series(equity, index=self.data.index)
        self.positions = pd.Series(positions, index=self.data.index)
        self.returns = pd.Series(returns, index=self.data.index)

    def performance_metrics(self):
        # Calculate Sharpe ratio, max drawdown, total return, win rate, etc.
        returns = self.equity_curve.pct_change().fillna(0)
        sharpe = np.sqrt(252) * returns.mean() / (returns.std() + 1e-8)
        cum_returns = self.equity_curve / self.initial_capital
        roll_max = cum_returns.cummax()
        drawdown = (cum_returns - roll_max) / roll_max
        max_dd = drawdown.min()
        total_return = cum_returns.iloc[-1] - 1
        # Trade stats
        trade_df = pd.DataFrame(self.trades)
        win_rate = np.nan
        avg_pnl = np.nan
        if not trade_df.empty and 'pnl' in trade_df.columns:
            wins = trade_df['pnl'] > 0
            win_rate = wins.sum() / len(trade_df)
            avg_pnl = trade_df['pnl'].mean()
        metrics = {
            'Sharpe Ratio': sharpe,
            'Max Drawdown': max_dd,
            'Total Return': total_return,
            'Win Rate': win_rate,
            'Avg Trade PnL': avg_pnl,
            'Num Trades': len(self.trades)
        }
        return metrics

    def plot_results(self):
        # Plot equity curve and positions
        fig, ax1 = plt.subplots(figsize=(12,6))
        ax1.plot(self.equity_curve.index, self.equity_curve.values, label='Equity Curve')
        ax1.set_ylabel('Equity')
        ax2 = ax1.twinx()
        ax2.plot(self.positions.index, self.positions.values, color='orange', alpha=0.3, label='Position')
        ax2.set_ylabel('Position')
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        plt.title('MicroEdge Momentum Strategy Results')
        plt.show()

# Sample data generation for TSLA-like price series
def generate_sample_tsla_data(n_days=252*2, seed=42):
    np.random.seed(seed)
    dates = pd.bdate_range('2022-01-01', periods=n_days)
    # Simulate log returns with volatility
    mu = 0.0005
    sigma = 0.025
    log_rets = np.random.normal(mu, sigma, size=n_days)
    price = 700 * np.exp(np.cumsum(log_rets))
    df = pd.DataFrame(" + str('Close': price) + ", index=dates)
    return df

# Main execution block
if __name__ == '__main__':
    # Generate sample data
    data = generate_sample_tsla_data()
    # Initialize strategy
    strategy = MicroEdgeMomentumStrategy(data, initial_capital=100000, position_size_pct=0.1, slippage=0.0005, commission=1.0)
    # Generate signals
    strategy.generate_signals()
    # Run backtest
    strategy.backtest()
    # Show performance metrics
    metrics = strategy.performance_metrics()
    print("Performance Metrics:")
    for k, v in metrics.items():
        print("%s: %s" % (k, v))
    # Plot results
    strategy.plot_results()
    # Print trade log summary
    print("Trade Log (first 5 trades):")
    for trade in strategy.trades[:5]:
        print(trade)

# =============================================================================
# STRATEGY EXECUTION AND TESTING
# =============================================================================

if __name__ == "__main__":
    # Example usage and testing
    strategy = MicroEdgeStrategy()
    print(f"Strategy '{strategyName}' initialized successfully!")
    
    # Example data loading
    symbols = ['SPY', 'QQQ', 'IWM']
    start_date = '2020-01-01'
    end_date = '2023-12-31'
    
    print(f"Loading data for symbols: {symbols}")
    data = strategy.load_data(symbols, start_date, end_date)
    
    if data is not None:
        print(f"Data loaded successfully. Shape: {data.shape}")
        print("Strategy ready for backtesting!")
    else:
        print("Failed to load data. Check your internet connection.")
