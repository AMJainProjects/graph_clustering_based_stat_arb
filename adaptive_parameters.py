import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from hmmlearn import hmm
from statsmodels.tsa.stattools import acf
from scipy import stats

class AdaptiveParameterManager:
    """
    Manages dynamic parameter selection based on market conditions.
    """
    
    def __init__(
            self,
            returns: pd.DataFrame,
            lookback_min: int = 5,
            lookback_max: int = 60,
            rebalance_min: int = 1,
            rebalance_max: int = 10,
            threshold_min: float = 0.0,
            threshold_max: float = 0.01,
            correlation_window_min: int = 10,
            correlation_window_max: int = 60,
            n_regimes: int = 3
    ):
        """
        Initializes the adaptive parameter manager.
        
        Args:
            returns: DataFrame with stock returns
            lookback_min: Minimum lookback window
            lookback_max: Maximum lookback window
            rebalance_min: Minimum rebalance period
            rebalance_max: Maximum rebalance period
            threshold_min: Minimum threshold for winner/loser classification
            threshold_max: Maximum threshold for winner/loser classification
            correlation_window_min: Minimum correlation window
            correlation_window_max: Maximum correlation window
            n_regimes: Number of market regimes to detect
        """
        self.returns = returns
        self.lookback_min = lookback_min
        self.lookback_max = lookback_max
        self.rebalance_min = rebalance_min
        self.rebalance_max = rebalance_max
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max
        self.correlation_window_min = correlation_window_min
        self.correlation_window_max = correlation_window_max
        self.n_regimes = n_regimes
        
        # Market proxy (assuming SPY is in the returns)
        if 'SPY' in returns.columns:
            self.market_returns = returns['SPY']
        else:
            # Use mean of all stocks as market proxy if SPY not available
            self.market_returns = returns.mean(axis=1)
        
        # Initialize regime detection
        self.regimes = None
        self.regime_params = None
        self.hmm_model = None
        
        # Detect regimes
        self._detect_regimes()
    
    def _detect_regimes(self) -> None:
        """
        Detects market regimes using Hidden Markov Model.
        """
        try:
            # Reshape data for HMM
            X = self.market_returns.values.reshape(-1, 1)
            
            # Initialize and fit HMM
            model = hmm.GaussianHMM(
                n_components=self.n_regimes,
                covariance_type="full",
                n_iter=1000,
                random_state=42,
            )
            
            model.fit(X)
            
            # Get regime classifications
            self.regimes = model.predict(X)
            
            # Identify regime characteristics
            self.regime_params = {}
            for i in range(self.n_regimes):
                mask = self.regimes == i
                regime_data = self.market_returns[mask]
                
                self.regime_params[i] = {
                    'mean': regime_data.mean(),
                    'std': regime_data.std(),
                    'frequency': mask.mean(),
                    'count': mask.sum()
                }
            
            # Store the model
            self.hmm_model = model
            
        except Exception as e:
            print(f"Error detecting regimes: {e}")
            # Fall back to assuming a single regime
            self.regimes = np.zeros(len(self.market_returns))
            self.regime_params = {
                0: {
                    'mean': self.market_returns.mean(),
                    'std': self.market_returns.std(),
                    'frequency': 1.0,
                    'count': len(self.market_returns)
                }
            }
    
    def get_current_regime(self, date: Optional[pd.Timestamp] = None) -> int:
        """
        Gets the regime for the current or specified date.
        
        Args:
            date: Optional date to get regime for
            
        Returns:
            Regime identifier
        """
        if date is None:
            # Use last regime if date not specified
            return self.regimes[-1]
        
        # Find the index of the specified date
        try:
            date_idx = self.returns.index.get_loc(date)
            return self.regimes[date_idx]
        except (KeyError, IndexError):
            # Return the most recent regime if date not found
            return self.regimes[-1]
    
    def calculate_optimal_lookback(self, method: str = 'volatility', date: Optional[pd.Timestamp] = None) -> int:
        """
        Calculates optimal lookback window based on market conditions.
        
        Args:
            method: Method to use ('volatility', 'autocorrelation', 'regime')
            date: Optional date to calculate for
            
        Returns:
            Optimal lookback window size
        """
        if method == 'volatility':
            # Calculate optimal window based on recent vs. long-term volatility
            # Get data up to the specified date or all data if not specified
            if date is not None:
                end_idx = self.returns.index.get_loc(date)
                market_returns = self.market_returns.iloc[:end_idx+1]
            else:
                market_returns = self.market_returns
            
            # Need at least 20 days of data
            if len(market_returns) < 20:
                return self.lookback_min
            
            # Calculate volatility ratios
            recent_vol = market_returns.iloc[-20:].std()
            long_vol = market_returns.iloc[-120:].std() if len(market_returns) >= 120 else market_returns.std()
            
            volatility_ratio = recent_vol / long_vol
            
            # High volatility -> shorter lookback, low volatility -> longer lookback
            if volatility_ratio > 1.5:  # High recent volatility
                return self.lookback_min
            elif volatility_ratio < 0.7:  # Low recent volatility
                return self.lookback_max
            else:
                # Linear scaling between min and max
                vol_scale = (volatility_ratio - 0.7) / 0.8  # Normalize to 0-1
                return int(self.lookback_min + (self.lookback_max - self.lookback_min) * (1 - vol_scale))
        
        elif method == 'autocorrelation':
            # Calculate optimal window based on autocorrelation
            # Get data up to the specified date or all data if not specified
            if date is not None:
                end_idx = self.returns.index.get_loc(date)
                market_returns = self.market_returns.iloc[:end_idx+1]
            else:
                market_returns = self.market_returns
            
            # Need at least lookback_max days of data
            if len(market_returns) < self.lookback_max:
                return self.lookback_min
            
            # Calculate autocorrelation for different lags
            acf_values = []
            for lag in range(self.lookback_min, self.lookback_max + 1):
                # Calculate autocorrelation at this lag
                try:
                    acf_val = acf(market_returns, nlags=lag)[-1]
                    acf_values.append(abs(acf_val))  # Use absolute value of autocorrelation
                except:
                    acf_values.append(0)
            
            # Return lag with highest autocorrelation
            return self.lookback_min + np.argmax(acf_values)
        
        elif method == 'regime':
            # Get parameters based on current regime
            current_regime = self.get_current_regime(date)
            regime_std = self.regime_params[current_regime]['std']
            
            # Map regime volatility to lookback window
            # Higher volatility -> shorter lookback
            all_stds = [params['std'] for params in self.regime_params.values()]
            min_std, max_std = min(all_stds), max(all_stds)
            
            if max_std == min_std:
                # No difference in volatility between regimes
                return (self.lookback_min + self.lookback_max) // 2
            
            # Normalize std to 0-1 scale and invert (higher vol -> lower on 0-1 scale)
            normalized_std = 1 - (regime_std - min_std) / (max_std - min_std)
            
            # Scale to lookback range
            lookback = int(self.lookback_min + normalized_std * (self.lookback_max - self.lookback_min))
            
            return lookback
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def calculate_optimal_rebalance_period(self, method: str = 'volatility', date: Optional[pd.Timestamp] = None) -> int:
        """
        Calculates optimal rebalance period based on market conditions.
        
        Args:
            method: Method to use ('volatility', 'turnover', 'regime')
            date: Optional date to calculate for
            
        Returns:
            Optimal rebalance period
        """
        if method == 'volatility':
            # Higher volatility -> more frequent rebalancing
            # Get the optimal lookback window using volatility method
            lookback = self.calculate_optimal_lookback('volatility', date)
            
            # Map lookback to rebalance period inversely
            # lookback_min -> rebalance_min, lookback_max -> rebalance_max
            normalized_lookback = (lookback - self.lookback_min) / (self.lookback_max - self.lookback_min)
            
            # Invert the scale: shorter lookback -> shorter rebalance period
            rebalance_period = int(self.rebalance_min + (1 - normalized_lookback) * 
                               (self.rebalance_max - self.rebalance_min))
            
            return rebalance_period
        
        elif method == 'regime':
            # Get parameters based on current regime
            current_regime = self.get_current_regime(date)
            regime_std = self.regime_params[current_regime]['std']
            
            # Map regime volatility to rebalance period
            # Higher volatility -> more frequent rebalancing
            all_stds = [params['std'] for params in self.regime_params.values()]
            min_std, max_std = min(all_stds), max(all_stds)
            
            if max_std == min_std:
                # No difference in volatility between regimes
                return (self.rebalance_min + self.rebalance_max) // 2
            
            # Normalize std to 0-1 scale and invert (higher vol -> lower value)
            normalized_std = 1 - (regime_std - min_std) / (max_std - min_std)
            
            # Scale to rebalance range
            rebalance_period = int(self.rebalance_min + normalized_std * 
                                 (self.rebalance_max - self.rebalance_min))
            
            return rebalance_period
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def calculate_optimal_threshold(self, method: str = 'volatility', date: Optional[pd.Timestamp] = None) -> float:
        """
        Calculates optimal threshold for winner/loser classification.
        
        Args:
            method: Method to use ('volatility', 'dispersion', 'regime')
            date: Optional date to calculate for
            
        Returns:
            Optimal threshold value
        """
        if method == 'volatility':
            # Higher volatility -> higher threshold
            if date is not None:
                end_idx = self.returns.index.get_loc(date)
                market_returns = self.market_returns.iloc[:end_idx+1]
            else:
                market_returns = self.market_returns
            
            # Calculate recent volatility (20-day)
            recent_vol = market_returns.iloc[-20:].std() if len(market_returns) >= 20 else market_returns.std()
            
            # Scale volatility to threshold range
            # Typical range: market vol 0.01 (1%) -> threshold 0.005 (0.5%)
            vol_scaling_factor = 0.5  # Adjust as needed
            threshold = recent_vol * vol_scaling_factor
            
            # Constrain to min-max range
            return max(self.threshold_min, min(self.threshold_max, threshold))
        
        elif method == 'dispersion':
            # Higher cross-sectional dispersion -> higher threshold
            if date is not None:
                end_idx = self.returns.index.get_loc(date)
                stock_returns = self.returns.iloc[:end_idx+1]
            else:
                stock_returns = self.returns
            
            # Calculate cross-sectional dispersion (std across stocks)
            # Use last 20 days if available
            if len(stock_returns) >= 20:
                recent_returns = stock_returns.iloc[-20:]
                # Calculate std across stocks for each day, then average
                dispersion = recent_returns.std(axis=1).mean()
            else:
                dispersion = stock_returns.std(axis=1).mean()
            
            # Scale dispersion to threshold range
            # Typical range: dispersion 0.02 (2%) -> threshold 0.005 (0.5%)
            dispersion_scaling_factor = 0.25  # Adjust as needed
            threshold = dispersion * dispersion_scaling_factor
            
            # Constrain to min-max range
            return max(self.threshold_min, min(self.threshold_max, threshold))
        
        elif method == 'regime':
            # Get parameters based on current regime
            current_regime = self.get_current_regime(date)
            regime_std = self.regime_params[current_regime]['std']
            
            # Map regime volatility to threshold
            # Higher volatility -> higher threshold
            scaling_factor = 0.5  # Adjust as needed
            threshold = regime_std * scaling_factor
            
            # Constrain to min-max range
            return max(self.threshold_min, min(self.threshold_max, threshold))
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def calculate_optimal_correlation_window(self, method: str = 'volatility', date: Optional[pd.Timestamp] = None) -> int:
        """
        Calculates optimal window for correlation matrix calculation.
        
        Args:
            method: Method to use ('volatility', 'stability', 'regime')
            date: Optional date to calculate for
            
        Returns:
            Optimal correlation window size
        """
        if method == 'volatility':
            # Get the optimal lookback window first
            lookback = self.calculate_optimal_lookback('volatility', date)
            
            # Scale the correlation window based on the lookback
            # Longer lookback -> longer correlation window
            normalized_lookback = (lookback - self.lookback_min) / (self.lookback_max - self.lookback_min)
            
            correlation_window = int(self.correlation_window_min + normalized_lookback * 
                                  (self.correlation_window_max - self.correlation_window_min))
            
            return correlation_window
        
        elif method == 'regime':
            # Get parameters based on current regime
            current_regime = self.get_current_regime(date)
            regime_std = self.regime_params[current_regime]['std']
            
            # Map regime volatility to correlation window
            # Higher volatility -> shorter correlation window
            all_stds = [params['std'] for params in self.regime_params.values()]
            min_std, max_std = min(all_stds), max(all_stds)
            
            if max_std == min_std:
                # No difference in volatility between regimes
                return (self.correlation_window_min + self.correlation_window_max) // 2
            
            # Normalize std to 0-1 scale and invert (higher vol -> lower on 0-1 scale)
            normalized_std = 1 - (regime_std - min_std) / (max_std - min_std)
            
            # Scale to correlation window range
            correlation_window = int(self.correlation_window_min + normalized_std * 
                                  (self.correlation_window_max - self.correlation_window_min))
            
            return correlation_window
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def get_regime_specific_parameters(self, date: Optional[pd.Timestamp] = None) -> Dict:
        """
        Gets a complete set of optimal parameters for the current market regime.
        
        Args:
            date: Optional date to get parameters for
            
        Returns:
            Dictionary with optimal parameter values
        """
        # Get current regime
        current_regime = self.get_current_regime(date)
        regime_std = self.regime_params[current_regime]['std']
        
        # Classify regime based on volatility
        all_stds = [params['std'] for params in self.regime_params.values()]
        all_stds.sort()
        
        if self.n_regimes <= 2:
            # Binary classification
            if regime_std >= np.median(all_stds):
                regime_type = 'high_volatility'
            else:
                regime_type = 'low_volatility'
        else:
            # Multiple regimes
            if regime_std >= all_stds[-1] * 0.9:  # Top 10% of volatility
                regime_type = 'high_volatility'
            elif regime_std <= all_stds[0] * 1.1:  # Bottom 10% of volatility
                regime_type = 'low_volatility'
            else:
                regime_type = 'medium_volatility'
        
        # Set parameters based on regime type
        if regime_type == 'high_volatility':
            return {
                'lookback_window': max(5, int(self.lookback_min * 1.5)),          # Short lookback
                'rebalance_period': self.rebalance_min,                           # Frequent rebalancing
                'threshold': min(self.threshold_max, regime_std * 0.5),           # Higher threshold
                'correlation_window': max(10, int(self.correlation_window_min * 1.5)),  # Shorter correlation window
                'stop_win_threshold': 0.03,                                       # Tighter stop-win
                'regime_type': regime_type,
                'regime_id': current_regime,
                'regime_volatility': regime_std
            }
        elif regime_type == 'medium_volatility':
            return {
                'lookback_window': (self.lookback_min + self.lookback_max) // 2,  # Medium lookback
                'rebalance_period': (self.rebalance_min + self.rebalance_max) // 2,  # Medium rebalancing
                'threshold': regime_std * 0.4,                                     # Medium threshold
                'correlation_window': (self.correlation_window_min + self.correlation_window_max) // 2,  # Medium correlation window
                'stop_win_threshold': 0.05,                                        # Medium stop-win
                'regime_type': regime_type,
                'regime_id': current_regime,
                'regime_volatility': regime_std
            }
        else:  # low_volatility
            return {
                'lookback_window': min(40, int(self.lookback_max * 0.8)),          # Longer lookback
                'rebalance_period': min(7, int(self.rebalance_max * 0.8)),         # Less frequent rebalancing
                'threshold': max(self.threshold_min, regime_std * 0.3),            # Lower threshold
                'correlation_window': min(60, int(self.correlation_window_max * 0.8)),  # Longer correlation window
                'stop_win_threshold': 0.07,                                        # Wider stop-win
                'regime_type': regime_type,
                'regime_id': current_regime,
                'regime_volatility': regime_std
            }
    
    def visualize_regimes(self):
        """
        Visualizes the detected market regimes.
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Create figure
            plt.figure(figsize=(12, 8))
            
            # Plot returns
            plt.subplot(2, 1, 1)
            for regime in range(self.n_regimes):
                mask = self.regimes == regime
                regime_returns = self.market_returns.copy()
                regime_returns[~mask] = np.nan
                plt.plot(regime_returns.index, regime_returns, 
                         label=f'Regime {regime} (vol={self.regime_params[regime]["std"]:.4f})')
            
            plt.title('Market Returns by Regime')
            plt.ylabel('Return')
            plt.legend()
            plt.grid(True)
            
            # Plot regime probabilities if available
            if hasattr(self.hmm_model, 'predict_proba'):
                plt.subplot(2, 1, 2)
                probs = self.hmm_model.predict_proba(self.market_returns.values.reshape(-1, 1))
                
                # Create DataFrame for plotting
                prob_df = pd.DataFrame(
                    probs, 
                    index=self.market_returns.index,
                    columns=[f'Regime {i}' for i in range(self.n_regimes)]
                )
                
                # Plot stacked probabilities
                plt.stackplot(prob_df.index, prob_df.values.T, 
                             labels=[f'Regime {i}' for i in range(self.n_regimes)],
                             alpha=0.7)
                
                plt.title('Regime Probabilities')
                plt.ylabel('Probability')
                plt.legend(loc='upper left')
                plt.ylim(0, 1)
                plt.grid(True)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib and/or seaborn not available for visualization")

def detect_market_regime(
    returns: pd.DataFrame, 
    n_regimes: int = 3,
    market_ticker: str = 'SPY'
) -> Tuple[np.ndarray, Dict, hmm.GaussianHMM]:
    """
    Detects market regimes using Hidden Markov Model.
    
    Args:
        returns: DataFrame with stock returns
        n_regimes: Number of regimes to detect
        market_ticker: Ticker to use as market proxy
        
    Returns:
        Tuple of (regime_labels, regime_parameters, hmm_model)
    """
    # Get market returns
    if market_ticker in returns.columns:
        market_returns = returns[market_ticker]
    else:
        # Use mean of all stocks if market ticker not available
        market_returns = returns.mean(axis=1)
    
    # Reshape data for HMM
    X = market_returns.values.reshape(-1, 1)
    
    # Initialize and fit HMM
    model = hmm.GaussianHMM(
        n_components=n_regimes,
        covariance_type="full",
        n_iter=1000,
        random_state=42
    )
    
    model.fit(X)
    
    # Get regime classifications
    hidden_states = model.predict(X)
    
    # Identify regime characteristics
    regime_params = {}
    for i in range(n_regimes):
        mask = hidden_states == i
        regime_data = market_returns[mask]
        
        regime_params[i] = {
            'mean': regime_data.mean(),
            'std': regime_data.std(),
            'frequency': mask.mean(),
            'count': mask.sum()
        }
    
    return hidden_states, regime_params, model

def calculate_optimal_lookback_window(
    returns: pd.DataFrame,
    min_window: int = 5,
    max_window: int = 60,
    method: str = 'volatility'
) -> int:
    """
    Calculates optimal lookback window based on market conditions.
    
    Args:
        returns: DataFrame with stock returns
        min_window: Minimum window size
        max_window: Maximum window size
        method: Method to use ('volatility' or 'autocorrelation')
        
    Returns:
        Optimal window size
    """
    if method == 'volatility':
        # Get market returns (use SPY if available, otherwise mean of all stocks)
        if 'SPY' in returns.columns:
            market_returns = returns['SPY']
        else:
            market_returns = returns.mean(axis=1)
        
        # More volatile market = shorter lookback
        recent_vol = market_returns.iloc[-20:].std() if len(market_returns) >= 20 else market_returns.std()
        long_vol = market_returns.iloc[-120:].std() if len(market_returns) >= 120 else market_returns.std()
        
        volatility_ratio = recent_vol / long_vol
        
        # Scale ratio to window size
        if volatility_ratio > 1.5:  # High recent volatility
            window = min_window
        elif volatility_ratio < 0.7:  # Low recent volatility
            window = max_window
        else:
            # Linear scaling between min and max
            vol_scale = (volatility_ratio - 0.7) / 0.8  # Normalize to 0-1
            window = int(min_window + (max_window - min_window) * (1 - vol_scale))
            
        return window
    
    elif method == 'autocorrelation':
        # Get market returns
        if 'SPY' in returns.columns:
            market_returns = returns['SPY']
        else:
            market_returns = returns.mean(axis=1)
        
        # Find window with highest autocorrelation
        acf_values = []
        
        for window in range(min_window, max_window + 1):
            try:
                # Calculate autocorrelation
                acf_value = acf(market_returns, nlags=window)[-1]
                acf_values.append(abs(acf_value))  # Use absolute value
            except:
                acf_values.append(0)
            
        if not acf_values:
            return min_window
            
        # Return window with highest autocorrelation
        optimal_window = min_window + np.argmax(acf_values)
        return optimal_window
    
    else:
        raise ValueError(f"Unknown method: {method}")

if __name__ == "__main__":
    # Example usage
    import yfinance as yf
    
    # Fetch sample data
    tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'SPY']
    start_date = '2018-01-01'
    end_date = '2022-12-31'
    
    try:
        # Fetch price data
        prices = yf.download(tickers, start=start_date, end=end_date, progress=False)['Adj Close']
        
        # Calculate returns
        returns = prices.pct_change().dropna()
        
        # Initialize adaptive parameter manager
        param_manager = AdaptiveParameterManager(
            returns,
            lookback_min=5,
            lookback_max=40,
            rebalance_min=1,
            rebalance_max=7,
            n_regimes=3
        )
        
        # Get optimal parameters
        optimal_lookback = param_manager.calculate_optimal_lookback(method='volatility')
        optimal_rebalance = param_manager.calculate_optimal_rebalance_period(method='volatility')
        optimal_threshold = param_manager.calculate_optimal_threshold(method='volatility')
        optimal_corr_window = param_manager.calculate_optimal_correlation_window(method='volatility')
        
        print("Optimal Parameters (Volatility Method):")
        print(f"Lookback Window: {optimal_lookback}")
        print(f"Rebalance Period: {optimal_rebalance}")
        print(f"Threshold: {optimal_threshold:.6f}")
        print(f"Correlation Window: {optimal_corr_window}")
        
        # Get regime-specific parameters
        regime_params = param_manager.get_regime_specific_parameters()
        
        print("\nRegime-Specific Parameters:")
        for key, value in regime_params.items():
            if isinstance(value, float):
                print(f"{key}: {value:.6f}")
            else:
                print(f"{key}: {value}")
        
        # Visualize regimes
        param_manager.visualize_regimes()
        
    except Exception as e:
        print(f"Error in example: {e}")
        
        # Create mock data if fetching fails
        import numpy as np
        import pandas as pd
        from datetime import datetime, timedelta
        
        # Create mock data
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        mock_returns = pd.DataFrame(
            np.random.normal(0.0005, 0.01, (len(dates), len(tickers))),
            index=dates,
            columns=tickers
        )
        
        print("Created mock data for demonstration.")
        
        # Initialize adaptive parameter manager with mock data
        param_manager = AdaptiveParameterManager(
            mock_returns,
            lookback_min=5,
            lookback_max=40,
            rebalance_min=1,
            rebalance_max=7,
            n_regimes=3
        )
        
        # Get optimal parameters
        optimal_lookback = param_manager.calculate_optimal_lookback(method='volatility')
        optimal_rebalance = param_manager.calculate_optimal_rebalance_period(method='volatility')
        optimal_threshold = param_manager.calculate_optimal_threshold(method='volatility')
        optimal_corr_window = param_manager.calculate_optimal_correlation_window(method='volatility')
        
        print("Optimal Parameters (Volatility Method):")
        print(f"Lookback Window: {optimal_lookback}")
        print(f"Rebalance Period: {optimal_rebalance}")
        print(f"Threshold: {optimal_threshold:.6f}")
        print(f"Correlation Window: {optimal_corr_window}")
