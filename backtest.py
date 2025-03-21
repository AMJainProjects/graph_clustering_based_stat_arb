import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable
import time
from datetime import datetime
import copy
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
import random

from data_preprocessing import prepare_stock_data, prepare_multifactor_data
from correlation_matrix import compute_correlation_matrix
from clustering import cluster_stocks, get_clusters_dict, determine_clusters_variance, determine_clusters_mp
from portfolio_construction import (
    identify_winners_losers,
    construct_arbitrage_portfolios,
    calculate_portfolio_returns
)
from utils import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_annualized_return
)
from adaptive_parameters import AdaptiveParameterManager, calculate_optimal_lookback_window


class StatisticalArbitrageBacktest:
    """
    Backtests a statistical arbitrage strategy based on correlation matrix clustering.
    """

    def __init__(
            self,
            tickers: List[str],
            start_date: str,
            end_date: str,
            clustering_method: str = 'sponge_sym',
            beta_window: int = 120,
            correlation_window: int = 40,
            lookback_window: int = 30,
            rebalance_period: int = 3,
            threshold: float = 0.005,
            stop_win_threshold: Optional[float] = 0.05,
            market_cap_percentile: float = 0.75,
            n_clusters_method: Optional[str] = None,
            fixed_n_clusters: Optional[int] = None,
            factor_model: str = 'capm',
            use_dynamic_parameters: bool = False,
            use_alternative_data: bool = False
    ):
        """
        Initializes the backtest.

        Args:
            tickers: List of stock ticker symbols
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            clustering_method: Method for clustering ('spectral', 'signed_laplacian_sym', etc.)
            beta_window: Window for beta calculation in days
            correlation_window: Window for correlation calculation in days
            lookback_window: Window for identifying winners/losers in days
            rebalance_period: Number of days between portfolio rebalancing
            threshold: Threshold for classifying as winner/loser
            stop_win_threshold: Stop-win threshold (or None to disable)
            market_cap_percentile: Percentile for market cap filtering
            n_clusters_method: Method for determining number of clusters ('mp' or 'var')
            fixed_n_clusters: Fixed number of clusters (or None to determine dynamically)
            factor_model: Model for residual calculation ('capm', 'fama_french', 'pca')
            use_dynamic_parameters: Whether to use adaptive parameters
            use_alternative_data: Whether to use alternative data sources
        """
        self.tickers = tickers
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.clustering_method = clustering_method
        self.beta_window = beta_window
        self.correlation_window = correlation_window
        self.lookback_window = lookback_window
        self.rebalance_period = rebalance_period
        self.threshold = threshold
        self.stop_win_threshold = stop_win_threshold
        self.market_cap_percentile = market_cap_percentile
        self.n_clusters_method = n_clusters_method
        self.fixed_n_clusters = fixed_n_clusters
        self.factor_model = factor_model
        self.use_dynamic_parameters = use_dynamic_parameters
        self.use_alternative_data = use_alternative_data

        # These will be populated during backtest
        self.returns = None
        self.betas = None
        self.residual_returns = None
        self.factor_returns = None
        self.factor_data = None
        self.portfolio_returns = None
        self.cluster_history = []
        self.portfolio_history = []
        self.parameter_history = []
        self.adaptive_param_manager = None
        self.alternative_data = {}

    def prepare_data(self):
        """
        Prepares the data for backtesting.
        """
        if self.factor_model == 'capm':
            # Use traditional CAPM approach
            returns, betas, residual_returns = prepare_stock_data(
                self.tickers,
                self.start_date.strftime('%Y-%m-%d'),
                self.end_date.strftime('%Y-%m-%d'),
                beta_window=self.beta_window,
                market_cap_percentile=self.market_cap_percentile
            )

            self.returns = returns
            self.betas = betas
            self.residual_returns = residual_returns
            self.factor_returns = pd.DataFrame({'MKT': returns['SPY']})
            self.factor_data = {'betas': betas}
        elif self.factor_model == 'fama_french':
            # Use multi-factor model
            returns, factor_returns, residual_returns, factor_data = prepare_multifactor_data(
                self.tickers,
                self.start_date.strftime('%Y-%m-%d'),
                self.end_date.strftime('%Y-%m-%d'),
                factor_model=self.factor_model,
                rolling_window=self.beta_window,
                market_cap_percentile=self.market_cap_percentile
            )

            self.returns = returns
            self.residual_returns = residual_returns
            self.factor_returns = factor_returns
            self.factor_data = factor_data

            # Extract betas for compatibility
            if 'betas' in factor_data:
                # Convert dict of DataFrames to DataFrame
                if isinstance(factor_data['betas'], dict):
                    # Use the first factor as a proxy
                    factor = list(factor_data['betas'][list(factor_data['betas'].keys())[0]].columns)[0]
                    betas = pd.DataFrame(
                        index=residual_returns.index,
                        columns=residual_returns.columns
                    )
                    for ticker in residual_returns.columns:
                        if ticker in factor_data['betas']:
                            betas[ticker] = factor_data['betas'][ticker][factor]
                    self.betas = betas
                else:
                    self.betas = factor_data['betas']

        # Initialize adaptive parameter manager if requested
        if self.use_dynamic_parameters:
            self.adaptive_param_manager = AdaptiveParameterManager(
                self.returns,
                lookback_min=3,
                lookback_max=20,
                rebalance_min=1,
                rebalance_max=7,
                threshold_min=0.0,
                threshold_max=0.01,
                correlation_window_min=10,
                correlation_window_max=60,
                n_regimes=3
            )

    def add_alternative_data(self, data_type: str, data: Dict):
        """
        Adds alternative data to the backtest.

        Args:
            data_type: Type of alternative data ('sentiment', 'options', 'order_flow')
            data: Dictionary with alternative data
        """
        self.alternative_data[data_type] = data
        self.use_alternative_data = True

    def _get_adaptive_parameters(self, date: pd.Timestamp) -> Dict:
        """
        Gets optimal parameters for the current date.

        Args:
            date: Current date

        Returns:
            Dictionary with optimal parameters
        """
        if self.adaptive_param_manager is None:
            # Return default parameters
            return {
                'lookback_window': self.lookback_window,
                'rebalance_period': self.rebalance_period,
                'threshold': self.threshold,
                'correlation_window': self.correlation_window,
                'stop_win_threshold': self.stop_win_threshold
            }

        # Get optimal parameters
        return self.adaptive_param_manager.get_regime_specific_parameters(date)

    def _apply_sentiment_data(self, returns: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
        """
        Adjusts returns based on sentiment data.

        Args:
            returns: DataFrame with returns
            date: Current date

        Returns:
            Adjusted returns
        """
        if 'sentiment' not in self.alternative_data:
            return returns

        sentiment_data = self.alternative_data['sentiment']
        if date not in sentiment_data.index:
            return returns

        # Get sentiment scores for current date
        current_sentiment = sentiment_data.loc[date]

        # Adjust returns based on sentiment (simple approach)
        adjusted_returns = returns.copy()

        for ticker in returns.columns:
            if ticker in current_sentiment:
                sentiment_score = current_sentiment[ticker]

                # Apply small adjustment based on sentiment
                adjustment = sentiment_score * 0.05  # Scale factor
                adjusted_returns[ticker] = returns[ticker] * (1 + adjustment)

        return adjusted_returns

    def _apply_options_signals(self, winners: Dict, losers: Dict, date: pd.Timestamp) -> Tuple[Dict, Dict]:
        """
        Adjusts winners and losers based on options signals.

        Args:
            winners: Dictionary of winner stocks by cluster
            losers: Dictionary of loser stocks by cluster
            date: Current date

        Returns:
            Adjusted winners and losers
        """
        if 'options' not in self.alternative_data:
            return winners, losers

        options_signals = self.alternative_data['options']

        # Copy to avoid modifying originals
        adjusted_winners = copy.deepcopy(winners)
        adjusted_losers = copy.deepcopy(losers)

        for cluster_id in winners.keys():
            cluster_winners = winners[cluster_id]
            cluster_losers = losers[cluster_id]

            for ticker in cluster_winners:
                if ticker in options_signals:
                    # Get signals for this ticker
                    signals = options_signals[ticker]

                    # Check signals (example: high put/call ratio suggests bearishness)
                    if 'put_call_ratio' in signals and signals['put_call_ratio'] > 1.5:
                        # Move from winners to losers (bearish signal)
                        if ticker in adjusted_winners[cluster_id]:
                            adjusted_winners[cluster_id].remove(ticker)
                        if ticker not in adjusted_losers[cluster_id]:
                            adjusted_losers[cluster_id].append(ticker)

            for ticker in cluster_losers:
                if ticker in options_signals:
                    # Get signals for this ticker
                    signals = options_signals[ticker]

                    # Check signals (example: very negative skew suggests potential rebound)
                    if 'skew_spread' in signals and signals['skew_spread'] < -10:
                        # Move from losers to winners (bullish signal)
                        if ticker in adjusted_losers[cluster_id]:
                            adjusted_losers[cluster_id].remove(ticker)
                        if ticker not in adjusted_winners[cluster_id]:
                            adjusted_winners[cluster_id].append(ticker)

        return adjusted_winners, adjusted_losers

    def _apply_order_flow_data(self, threshold: float, date: pd.Timestamp) -> Dict[str, float]:
        """
        Adjusts threshold based on order flow data.

        Args:
            threshold: Base threshold value
            date: Current date

        Returns:
            Dictionary mapping tickers to adjusted thresholds
        """
        if 'order_flow' not in self.alternative_data:
            return {ticker: threshold for ticker in self.returns.columns}

        order_flow_data = self.alternative_data['order_flow']

        # Initialize adjusted thresholds
        adjusted_thresholds = {}

        for ticker in self.returns.columns:
            if ticker in order_flow_data:
                # Get recent order flow imbalance
                imbalance = order_flow_data[ticker]

                # Find nearest date
                if isinstance(imbalance.index, pd.DatetimeIndex):
                    # Find nearest date
                    dates_before = imbalance.index[imbalance.index <= date]
                    if len(dates_before) > 0:
                        nearest_date = dates_before[-1]
                        recent_imbalance = imbalance.loc[nearest_date]

                        # Adjust threshold based on imbalance
                        # Strong buying pressure -> raise threshold for shorting
                        # Strong selling pressure -> raise threshold for longing
                        adjustment = 0.002 * recent_imbalance  # Scaling factor

                        adjusted_thresholds[ticker] = threshold + adjustment
                    else:
                        adjusted_thresholds[ticker] = threshold
                else:
                    adjusted_thresholds[ticker] = threshold
            else:
                adjusted_thresholds[ticker] = threshold

        return adjusted_thresholds

    def run_backtest(self):
        """
        Runs the statistical arbitrage backtest.

        Returns:
            DataFrame with daily portfolio returns
        """
        if self.returns is None or self.residual_returns is None:
            self.prepare_data()

        # Initialize portfolio returns DataFrame
        all_portfolio_returns = pd.DataFrame(index=self.residual_returns.index)
        all_portfolio_returns['Combined'] = 0.0

        # Initialize additional tracking variables
        current_portfolio = {}  # Tracks current holdings
        previous_portfolio = None  # For transaction cost calculation
        last_rebalance_date = self.residual_returns.index[0]
        stop_win_triggered = False

        # Track transaction costs
        self.transaction_costs = pd.DataFrame(index=self.residual_returns.index)
        self.transaction_costs['TC'] = 0.0
        self.total_transaction_cost = 0.0

        for i, date in enumerate(self.residual_returns.index):
            # Get optimal parameters if using dynamic parameters
            if self.use_dynamic_parameters:
                params = self._get_adaptive_parameters(date)
                current_lookback = params['lookback_window']
                current_rebalance = params['rebalance_period']
                current_threshold = params['threshold']
                current_corr_window = params['correlation_window']
                current_stop_win = params.get('stop_win_threshold', self.stop_win_threshold)

                # Store parameter history
                self.parameter_history.append({
                    'date': date,
                    'lookback_window': current_lookback,
                    'rebalance_period': current_rebalance,
                    'threshold': current_threshold,
                    'correlation_window': current_corr_window,
                    'stop_win_threshold': current_stop_win,
                    'regime_type': params.get('regime_type', 'unknown'),
                    'regime_id': params.get('regime_id', 0)
                })
            else:
                current_lookback = self.lookback_window
                current_rebalance = self.rebalance_period
                current_threshold = self.threshold
                current_corr_window = self.correlation_window
                current_stop_win = self.stop_win_threshold

            # Check if we need to rebalance
            days_since_rebalance = (date - last_rebalance_date).days

            if i == 0 or days_since_rebalance >= current_rebalance or stop_win_triggered:
                # We need at least correlation_window days of data
                if i < current_corr_window:
                    continue

                # Update rebalance tracking
                last_rebalance_date = date
                stop_win_triggered = False

                # Get data for correlation matrix
                lookback_start = max(0, i - current_corr_window)
                lookback_data = self.residual_returns.iloc[lookback_start:i]

                # Calculate correlation matrix
                correlation_matrix = compute_correlation_matrix(lookback_data)
                correlation_matrix = correlation_matrix.drop(labels='SPY', errors='ignore')
                correlation_matrix = correlation_matrix.drop(index=['SPY'], axis=1, errors='ignore')
                correlation_matrix = correlation_matrix.drop(columns=['SPY'], axis=0, errors='ignore')

                # Determine number of clusters
                n_clusters = self.fixed_n_clusters
                if n_clusters is None:
                    if self.n_clusters_method == 'mp':
                        # Use Marchenko-Pastur method
                        n_clusters = cluster_stocks(
                            correlation_matrix,
                            lookback_periods=current_corr_window
                        )[1]
                    elif self.n_clusters_method == 'var':
                        # Use variance explained method
                        n_clusters = cluster_stocks(
                            correlation_matrix
                        )[1]
                    else:
                        n_clusters_mp = determine_clusters_mp(correlation_matrix, self.correlation_window)
                        n_clusters_var = determine_clusters_variance(correlation_matrix, variance_explained=0.9)
                        n_clusters = int((n_clusters_mp + n_clusters_var) / 2)

                # Cluster stocks
                labels, _ = cluster_stocks(
                    correlation_matrix,
                    method='hierarchical',
                    n_clusters=n_clusters
                )

                clusters = get_clusters_dict(
                    labels, correlation_matrix.columns.tolist()
                )

                # Store cluster information
                self.cluster_history.append({
                    'date': date,
                    'n_clusters': n_clusters,
                    'clusters': clusters
                })

                # Get data for identifying winners/losers
                lookback_start = max(0, i - current_lookback)
                lookback_data = self.returns.iloc[lookback_start:i]

                # Apply sentiment data if available
                if self.use_alternative_data and 'sentiment' in self.alternative_data:
                    lookback_data = self._apply_sentiment_data(lookback_data, date)

                # Get ticker-specific thresholds if using order flow data
                if self.use_alternative_data and 'order_flow' in self.alternative_data:
                    thresholds = self._apply_order_flow_data(current_threshold, date)
                else:
                    thresholds = {ticker: current_threshold for ticker in self.returns.columns}

                thresholds = thresholds if (thresholds is not None and type(thresholds) != dict) else self.threshold

                # Identify winners and losers
                winners, losers = identify_winners_losers(
                    lookback_data,
                    clusters,
                    current_lookback,
                    thresholds
                )

                # Apply options signals if available
                if self.use_alternative_data and 'options' in self.alternative_data:
                    winners, losers = self._apply_options_signals(winners, losers, date)

                # Construct arbitrage portfolios
                current_portfolio = construct_arbitrage_portfolios(winners, losers)

                # Store portfolio information
                self.portfolio_history.append({
                    'date': date,
                    'portfolio': current_portfolio
                })

                # Update previous portfolio for transaction cost calculation
                previous_portfolio = current_portfolio.copy()

            # Calculate daily returns for current portfolio
            # Calculate daily returns for current portfolio
            if current_portfolio and i < len(self.residual_returns) - 1:
                # Get next day's returns
                next_day = self.residual_returns.index[i + 1]
                next_day_returns = self.residual_returns.loc[next_day]

                # Calculate portfolio returns with transaction costs
                day_returns = {}
                day_tc = {}

                # Only apply transaction costs on rebalance days
                apply_tc = (i == 0 or days_since_rebalance >= current_rebalance or stop_win_triggered)

                for cluster_id, portfolio in current_portfolio.items():
                    # Initialize return for this cluster
                    day_returns[f"Cluster_{cluster_id}"] = 0.0
                    day_tc[f"Cluster_{cluster_id}"] = 0.0

                    # Calculate transaction costs if applicable
                    if apply_tc and previous_portfolio is not None and cluster_id in previous_portfolio:
                        # Get previous portfolio for this cluster
                        prev_portfolio = previous_portfolio.get(cluster_id, {})

                        # Calculate turnover (sum of absolute weight changes)
                        turnover = 0.0
                        all_tickers = set(portfolio.keys()).union(set(prev_portfolio.keys()))

                        for ticker in all_tickers:
                            current_weight = portfolio.get(ticker, 0.0)
                            previous_weight = prev_portfolio.get(ticker, 0.0)
                            turnover += abs(current_weight - previous_weight)

                        # Apply transaction cost (5 basis points)
                        tc = turnover * 0.0005
                        day_tc[f"Cluster_{cluster_id}"] = tc

                        # Add safe handling for transaction costs
                        if next_day in self.transaction_costs.index:
                            self.transaction_costs.loc[next_day, 'TC'] += tc
                        else:
                            # Handle case where date doesn't exist in index
                            print(
                                f"Warning: Date {next_day} not found in transaction_costs index. Skipping TC recording.")

                        self.total_transaction_cost += tc

                    # Sum weighted returns
                    for ticker, weight in portfolio.items():
                        if ticker in next_day_returns.index:
                            day_returns[f"Cluster_{cluster_id}"] += weight * next_day_returns[ticker]

                    # Subtract transaction costs from returns
                    day_returns[f"Cluster_{cluster_id}"] -= day_tc[f"Cluster_{cluster_id}"]

                # Calculate combined return (equal weight across clusters)
                if day_returns:
                    day_returns['Combined'] = float(np.mean(list(day_returns.values())))
                    day_tc['Combined'] = float(np.mean(list(day_tc.values())))
                else:
                    day_returns['Combined'] = 0.0
                    day_tc['Combined'] = 0.0

                # Store in the all_portfolio_returns DataFrame
                for col, ret in day_returns.items():
                    if col in all_portfolio_returns.columns:
                        all_portfolio_returns.loc[next_day, col] = ret
                    else:
                        all_portfolio_returns[col] = 0.0
                        all_portfolio_returns.loc[next_day, col] = ret

                # Store transaction costs with safe handling
                for col, tc in day_tc.items():
                    tc_col = f"{col}_TC"
                    if tc_col not in all_portfolio_returns.columns:
                        all_portfolio_returns[tc_col] = 0.0

                    if next_day in all_portfolio_returns.index:
                        all_portfolio_returns.loc[next_day, tc_col] = tc

                # Check for stop-win
                if current_stop_win is not None:
                    # Calculate return since last rebalance
                    since_rebalance = all_portfolio_returns.loc[
                        all_portfolio_returns.index > last_rebalance_date,
                        'Combined'
                    ]

                    # Calculate cumulative return
                    cumulative_return = (1 + since_rebalance).prod() - 1

                    # Check if stop-win is triggered
                    if cumulative_return >= current_stop_win:
                        stop_win_triggered = True

        # Store the final portfolio returns
        self.portfolio_returns = all_portfolio_returns.fillna(0)

        return self.portfolio_returns

    def get_performance_metrics(self) -> pd.DataFrame:
        """
        Calculates and returns performance metrics for the backtest.

        Returns:
            DataFrame with performance metrics
        """
        if self.portfolio_returns is None:
            raise ValueError("Backtest has not been run yet.")

        # Initialize metrics DataFrame
        metrics = pd.DataFrame(index=self.portfolio_returns.columns)

        # Calculate metrics for each portfolio
        for col in self.portfolio_returns.columns:
            # Skip transaction cost columns
            if col.endswith('_TC'):
                continue

            returns = self.portfolio_returns[col]

            # Annualized return
            metrics.loc[col, 'Annualized Return (%)'] = calculate_annualized_return(returns) * 100

            # Sharpe ratio
            metrics.loc[col, 'Sharpe Ratio'] = calculate_sharpe_ratio(returns)

            # Sortino ratio
            metrics.loc[col, 'Sortino Ratio'] = calculate_sortino_ratio(returns)

            # Maximum drawdown
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.cummax()
            drawdown = (cumulative_returns / running_max) - 1
            metrics.loc[col, 'Maximum Drawdown (%)'] = drawdown.min() * 100

            # Win rate
            win_rate = (returns > 0).mean()
            metrics.loc[col, 'Win Rate (%)'] = win_rate * 100

            # Add transaction cost metrics if available
            tc_col = f"{col}_TC"
            if tc_col in self.portfolio_returns.columns:
                # Total transaction costs
                total_tc = self.portfolio_returns[tc_col].sum()
                metrics.loc[col, 'Total Transaction Costs (%)'] = total_tc * 100

                # Annualized transaction costs
                n_years = len(returns) / 252
                metrics.loc[col, 'Annualized Transaction Costs (%)'] = (total_tc / n_years) * 100

                # Average transaction costs per trade
                n_trades = (self.portfolio_returns[tc_col] > 0).sum()
                if n_trades > 0:
                    metrics.loc[col, 'Avg Transaction Cost per Trade (bps)'] = (total_tc / n_trades) * 10000
                else:
                    metrics.loc[col, 'Avg Transaction Cost per Trade (bps)'] = 0.0

                # Turnover (annualized)
                turnover = (total_tc / 0.0005) / n_years  # Assuming 5 bps cost
                metrics.loc[col, 'Annualized Turnover (x)'] = turnover

        return metrics

    def get_cumulative_returns(self) -> pd.DataFrame:
        """
        Calculates and returns cumulative returns for the backtest.

        Returns:
            DataFrame with cumulative returns
        """
        if self.portfolio_returns is None:
            raise ValueError("Backtest has not been run yet.")

        # Calculate cumulative returns
        cumulative_returns = (1 + self.portfolio_returns).cumprod() - 1

        return cumulative_returns

    def plot_performance(self, benchmark_returns: Optional[pd.Series] = None):
        """
        Plots performance of the strategy against a benchmark.

        Args:
            benchmark_returns: Optional benchmark returns series
        """
        import matplotlib.pyplot as plt

        if self.portfolio_returns is None:
            raise ValueError("Backtest has not been run yet.")

        # Calculate cumulative returns
        cumulative_returns = self.get_cumulative_returns()

        # Create plot
        plt.figure(figsize=(12, 6))

        # Plot strategy returns
        plt.plot(cumulative_returns.index, cumulative_returns['Combined'], label='Strategy')

        # Plot benchmark if provided
        if benchmark_returns is not None:
            # Align benchmark with strategy period
            benchmark_aligned = benchmark_returns.loc[
                (benchmark_returns.index >= cumulative_returns.index[0]) &
                (benchmark_returns.index <= cumulative_returns.index[-1])
                ]

            # Calculate cumulative benchmark returns
            cumulative_benchmark = (1 + benchmark_aligned).cumprod() - 1

            # Plot benchmark
            plt.plot(cumulative_benchmark.index, cumulative_benchmark, label='Benchmark')

        # Add labels and legend
        plt.title('Cumulative Returns')
        plt.xlabel('Date')
        plt.ylabel('Return')
        plt.legend()
        plt.grid(True)

        # Format y-axis as percentage
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

        plt.tight_layout()
        plt.show()

    def plot_adaptive_parameters(self):
        """
        Plots the history of adaptive parameters.
        """
        if not self.parameter_history:
            raise ValueError("No parameter history available.")

        # Convert parameter history to DataFrame
        param_df = pd.DataFrame(self.parameter_history)
        param_df.set_index('date', inplace=True)

        # Create plot
        fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True)

        # Plot lookback window
        axs[0].plot(param_df.index, param_df['lookback_window'], marker='o')
        axs[0].set_title('Adaptive Lookback Window')
        axs[0].set_ylabel('Days')
        axs[0].grid(True)

        # Plot rebalance period
        axs[1].plot(param_df.index, param_df['rebalance_period'], marker='o')
        axs[1].set_title('Adaptive Rebalance Period')
        axs[1].set_ylabel('Days')
        axs[1].grid(True)

        # Plot threshold
        axs[2].plot(param_df.index, param_df['threshold'], marker='o')
        axs[2].set_title('Adaptive Threshold')
        axs[2].set_ylabel('Threshold')
        axs[2].grid(True)

        # Plot correlation window
        axs[3].plot(param_df.index, param_df['correlation_window'], marker='o')
        axs[3].set_title('Adaptive Correlation Window')
        axs[3].set_ylabel('Days')
        axs[3].grid(True)

        # Customize x-axis
        plt.xlabel('Date')

        # Add regime background if available
        if 'regime_id' in param_df.columns:
            # Get unique regime IDs
            regimes = param_df['regime_id'].unique()
            colors = plt.cm.tab10(np.linspace(0, 1, len(regimes)))

            for ax in axs:
                for i, regime in enumerate(regimes):
                    regime_dates = param_df[param_df['regime_id'] == regime].index

                    # Group consecutive dates
                    groups = []
                    current_group = []

                    for j, date in enumerate(regime_dates):
                        if j == 0 or (date - regime_dates[j - 1]).days > 1:
                            if current_group:
                                groups.append(current_group)
                            current_group = [date]
                        else:
                            current_group.append(date)

                    if current_group:
                        groups.append(current_group)

                    # Add colored background for each group
                    for group in groups:
                        if len(group) > 1:
                            ax.axvspan(group[0], group[-1], alpha=0.2, color=colors[i])

        plt.tight_layout()
        plt.show()


# ==================== BACKTESTING EXTENSIONS ====================

def time_series_cross_validation(
        tickers: List[str],
        start_date: str,
        end_date: str,
        n_splits: int = 5,
        backtest_params: Optional[Dict] = None,
        **kwargs
) -> Dict:
    """
    Performs time-series cross-validation for the backtest.

    Args:
        tickers: List of stock tickers
        start_date: Start date for the data
        end_date: End date for the data
        n_splits: Number of cross-validation splits
        backtest_params: Dictionary with backtest parameters
        **kwargs: Additional parameters for backtest

    Returns:
        Dictionary with cross-validation results
    """
    # Fetch data for the entire period
    returns, betas, residual_returns = prepare_stock_data(
        tickers, start_date, end_date, **kwargs
    )

    # Create time series split
    tscv = TimeSeriesSplit(n_splits=n_splits)
    all_dates = residual_returns.index

    # Default parameters
    if backtest_params is None:
        backtest_params = {}

    # Store results
    results = []

    for train_idx, test_idx in tscv.split(all_dates):
        train_dates = all_dates[train_idx]
        test_dates = all_dates[test_idx]

        if len(train_dates) < 100 or len(test_dates) < 20:
            continue  # Skip if too little data

        try:
            print(f"Training period: {train_dates[0]} to {train_dates[-1]}")
            print(f"Testing period: {test_dates[0]} to {test_dates[-1]}")

            # Create training set
            train_returns = returns.loc[train_dates]
            train_residual_returns = residual_returns.loc[train_dates]

            # Initialize backtest with training data
            backtest = StatisticalArbitrageBacktest(
                tickers=tickers,
                start_date=train_dates[0].strftime('%Y-%m-%d'),
                end_date=train_dates[-1].strftime('%Y-%m-%d'),
                **backtest_params
            )

            # Directly set the data to avoid refetching
            backtest.returns = train_returns
            backtest.residual_returns = train_residual_returns

            # Run backtest on training data
            backtest.run_backtest()

            # Extract parameters for testing (simplified, would normally optimize)
            optimal_lookback = backtest.lookback_window
            optimal_rebalance = backtest.rebalance_period
            optimal_threshold = backtest.threshold

            # Create test backtest
            test_backtest = StatisticalArbitrageBacktest(
                tickers=tickers,
                start_date=test_dates[0].strftime('%Y-%m-%d'),
                end_date=test_dates[-1].strftime('%Y-%m-%d'),
                lookback_window=optimal_lookback,
                rebalance_period=optimal_rebalance,
                threshold=optimal_threshold,
                **{k: v for k, v in backtest_params.items() if
                   k not in ['lookback_window', 'rebalance_period', 'threshold']}
            )

            # Directly set the test data
            test_backtest.returns = returns.loc[returns.index.isin(train_dates) | returns.index.isin(test_dates)]
            test_backtest.residual_returns = residual_returns.loc[
                residual_returns.index.isin(train_dates) | residual_returns.index.isin(test_dates)]

            # Run backtest on test data
            test_returns = test_backtest.run_backtest()

            # Filter returns to just test period
            test_returns = test_returns.loc[test_dates]

            # Calculate performance metrics
            sharpe = calculate_sharpe_ratio(test_returns['Combined'])
            sortino = calculate_sortino_ratio(test_returns['Combined'])
            annualized_return = calculate_annualized_return(test_returns['Combined'])

            results.append({
                'train_start': train_dates[0],
                'train_end': train_dates[-1],
                'test_start': test_dates[0],
                'test_end': test_dates[-1],
                'parameters': {
                    'lookback_window': optimal_lookback,
                    'rebalance_period': optimal_rebalance,
                    'threshold': optimal_threshold
                },
                'sharpe_ratio': sharpe,
                'sortino_ratio': sortino,
                'annualized_return': annualized_return,
                'cumulative_return': (1 + test_returns['Combined']).prod() - 1
            })

        except Exception as e:
            print(f"Error in cross-validation fold: {e}")

    # Calculate summary statistics
    sharpe_values = [r['sharpe_ratio'] for r in results]
    sortino_values = [r['sortino_ratio'] for r in results]
    return_values = [r['annualized_return'] for r in results]
    cumulative_values = [r['cumulative_return'] for r in results]

    summary = {
        'results': results,
        'mean_sharpe': np.mean(sharpe_values),
        'std_sharpe': np.std(sharpe_values),
        'mean_sortino': np.mean(sortino_values),
        'std_sortino': np.std(sortino_values),
        'mean_annualized_return': np.mean(return_values),
        'std_annualized_return': np.std(return_values),
        'mean_cumulative_return': np.mean(cumulative_values),
        'std_cumulative_return': np.std(cumulative_values)
    }

    return summary


def monte_carlo_backtest(
        returns: pd.DataFrame,
        residual_returns: pd.DataFrame,
        n_simulations: int = 1000,
        backtest_params: Optional[Dict] = None,
        tickers: Optional[List[str]] = None,
        bootstrap_method: str = 'block'
) -> Dict:
    """
    Performs Monte Carlo simulation of the backtest.

    Args:
        returns: DataFrame with returns
        residual_returns: DataFrame with residual returns
        n_simulations: Number of simulations to run
        backtest_params: Dictionary with backtest parameters
        tickers: List of tickers (or None to use all in returns)
        bootstrap_method: Method for bootstrapping ('random', 'block')

    Returns:
        Dictionary with simulation results
    """
    if tickers is None:
        tickers = returns.columns.drop('SPY').tolist()

    # Default parameters
    if backtest_params is None:
        backtest_params = {
            'clustering_method': 'sponge_sym',
            'lookback_window': 5,
            'rebalance_period': 3,
            'threshold': 0.0
        }

    # Store results
    results = []

    for sim in range(n_simulations):
        try:
            if sim % 100 == 0:
                print(f"Running simulation {sim}/{n_simulations}...")

            # Generate bootstrapped samples
            if bootstrap_method == 'random':
                # Random sampling with replacement
                sample_indices = np.random.choice(
                    len(returns), size=len(returns), replace=True
                )
                bootstrap_returns = returns.iloc[sample_indices].copy()
                bootstrap_residual_returns = residual_returns.iloc[sample_indices].copy()
            elif bootstrap_method == 'block':
                # Block bootstrap (preserve autocorrelation)
                block_size = 20  # 20-day blocks
                n_blocks = len(returns) // block_size + 1

                # Sample blocks with replacement
                block_indices = np.random.choice(
                    len(returns) - block_size, size=n_blocks, replace=True
                )

                # Concatenate blocks
                sample_indices = []
                for block_start in block_indices:
                    sample_indices.extend(range(block_start, block_start + block_size))

                # Trim to original length
                sample_indices = sample_indices[:len(returns)]

                bootstrap_returns = returns.iloc[sample_indices].copy()
                bootstrap_residual_returns = residual_returns.iloc[sample_indices].copy()
            else:
                raise ValueError(f"Unknown bootstrap method: {bootstrap_method}")

            # Reset index to original dates
            bootstrap_returns.index = returns.index
            bootstrap_residual_returns.index = residual_returns.index

            # Initialize backtest
            backtest = StatisticalArbitrageBacktest(
                tickers=tickers,
                start_date=returns.index[0].strftime('%Y-%m-%d'),
                end_date=returns.index[-1].strftime('%Y-%m-%d'),
                **backtest_params
            )

            # Set the bootstrapped data
            backtest.returns = bootstrap_returns
            backtest.residual_returns = bootstrap_residual_returns

            # Run backtest
            backtest.run_backtest()

            # Calculate performance metrics
            metrics = backtest.get_performance_metrics()

            # Store key metrics
            results.append({
                'simulation': sim,
                'sharpe_ratio': metrics.loc['Combined', 'Sharpe Ratio'],
                'sortino_ratio': metrics.loc['Combined', 'Sortino Ratio'],
                'annualized_return': metrics.loc['Combined', 'Annualized Return (%)'] / 100,
                'max_drawdown': metrics.loc['Combined', 'Maximum Drawdown (%)'] / 100
            })

        except Exception as e:
            print(f"Error in simulation {sim}: {e}")

    # Create DataFrame of results
    results_df = pd.DataFrame(results)

    # Calculate summary statistics
    summary = {
        'results': results_df,
        'mean': results_df.mean(),
        'median': results_df.median(),
        'std': results_df.std(),
        '5th_percentile': results_df.quantile(0.05),
        '95th_percentile': results_df.quantile(0.95)
    }

    return summary


def walk_forward_optimization(
        tickers: List[str],
        start_date: str,
        end_date: str,
        initial_window: int = 252,
        step: int = 20,
        param_grid: Optional[Dict] = None,
        clustering_method: str = 'sponge_sym',
        factor_model: str = 'capm'
) -> Dict:
    """
    Performs walk-forward optimization of backtest parameters.

    Args:
        tickers: List of stock tickers
        start_date: Start date for the data
        end_date: End date for the data
        initial_window: Initial window size in days
        step: Step size for walk-forward in days
        param_grid: Dictionary with parameter grids to search
        clustering_method: Clustering method to use
        factor_model: Factor model to use

    Returns:
        Dictionary with optimization results
    """
    # Default parameter grid
    if param_grid is None:
        param_grid = {
            'lookback_window': [3, 5, 10, 15],
            'rebalance_period': [1, 3, 5, 7],
            'threshold': [0.0, 0.001, 0.003, 0.005]
        }

    # Fetch data for the entire period
    if factor_model == 'capm':
        returns, betas, residual_returns = prepare_stock_data(
            tickers, start_date, end_date
        )
        factor_returns = None
        factor_data = None
    else:
        returns, factor_returns, residual_returns, factor_data = prepare_multifactor_data(
            tickers, start_date, end_date, factor_model=factor_model
        )

    # Convert dates to numeric indices
    dates = residual_returns.index

    # Store results
    results = []

    # Iterate through time
    for train_end_idx in range(initial_window, len(dates), step):
        try:
            # Define train and test periods
            train_start_idx = 0
            train_end_idx = min(train_end_idx, len(dates) - 1)
            test_start_idx = train_end_idx
            test_end_idx = min(test_start_idx + step, len(dates) - 1)

            train_dates = dates[train_start_idx:train_end_idx]
            test_dates = dates[test_start_idx:test_end_idx]

            print(f"Training: {train_dates[0]} to {train_dates[-1]}")
            print(f"Testing: {test_dates[0]} to {test_dates[-1]}")

            # Skip if train or test is too short
            if len(train_dates) < 100 or len(test_dates) < 5:
                continue

            # Define parameter combinations
            param_combinations = []
            for lookback in param_grid['lookback_window']:
                for rebalance in param_grid['rebalance_period']:
                    for threshold in param_grid['threshold']:
                        param_combinations.append({
                            'lookback_window': lookback,
                            'rebalance_period': rebalance,
                            'threshold': threshold
                        })

            # Test each parameter combination on training set
            train_results = []

            for params in param_combinations:
                # Initialize backtest with training data
                backtest = StatisticalArbitrageBacktest(
                    tickers=tickers,
                    start_date=train_dates[0].strftime('%Y-%m-%d'),
                    end_date=train_dates[-1].strftime('%Y-%m-%d'),
                    clustering_method=clustering_method,
                    **params
                )

                # Set data directly
                backtest.returns = returns.loc[train_dates]
                backtest.residual_returns = residual_returns.loc[train_dates]
                if factor_returns is not None:
                    backtest.factor_returns = factor_returns.loc[factor_returns.index.isin(train_dates)]
                    backtest.factor_data = factor_data

                # Run backtest
                backtest.run_backtest()

                # Calculate performance metrics
                sharpe = calculate_sharpe_ratio(backtest.portfolio_returns['Combined'])
                sortino = calculate_sortino_ratio(backtest.portfolio_returns['Combined'])
                annualized_return = calculate_annualized_return(backtest.portfolio_returns['Combined'])

                train_results.append({
                    'params': params,
                    'sharpe_ratio': sharpe,
                    'sortino_ratio': sortino,
                    'annualized_return': annualized_return
                })

            # Find best parameters based on Sharpe ratio
            best_params = max(train_results, key=lambda x: x['sharpe_ratio'])['params']

            print(f"Best parameters: {best_params}")

            # Apply best parameters to test set
            test_backtest = StatisticalArbitrageBacktest(
                tickers=tickers,
                start_date=test_dates[0].strftime('%Y-%m-%d'),
                end_date=test_dates[-1].strftime('%Y-%m-%d'),
                clustering_method=clustering_method,
                **best_params
            )

            # Set data for test set (include some training data for lookback)
            lookback_start_idx = max(0, test_start_idx - 60)  # 60 days of lookback data
            lookback_dates = dates[lookback_start_idx:test_end_idx]

            test_backtest.returns = returns.loc[lookback_dates]
            test_backtest.residual_returns = residual_returns.loc[lookback_dates]
            if factor_returns is not None:
                test_backtest.factor_returns = factor_returns.loc[factor_returns.index.isin(lookback_dates)]
                test_backtest.factor_data = factor_data

            # Run backtest
            test_backtest.run_backtest()

            # Filter to just test period
            test_returns = test_backtest.portfolio_returns.loc[test_dates]

            # Calculate performance metrics
            sharpe = calculate_sharpe_ratio(test_returns['Combined'])
            sortino = calculate_sortino_ratio(test_returns['Combined'])
            annualized_return = calculate_annualized_return(test_returns['Combined'])
            cumulative_return = (1 + test_returns['Combined']).prod() - 1

            results.append({
                'train_start': train_dates[0],
                'train_end': train_dates[-1],
                'test_start': test_dates[0],
                'test_end': test_dates[-1],
                'best_params': best_params,
                'train_sharpe': best_params.get('sharpe_ratio', 0),
                'test_sharpe': sharpe,
                'test_sortino': sortino,
                'test_annual_return': annualized_return,
                'test_cumulative_return': cumulative_return
            })

        except Exception as e:
            print(f"Error in walk-forward step: {e}")

    # Create DataFrame of results
    results_df = pd.DataFrame(results)

    # Calculate aggregated performance
    if not results_df.empty and 'test_cumulative_return' in results_df.columns:
        # Calculate weighted average return (by number of days in test)
        weights = []
        for i in range(len(results_df)):
            test_days = (results_df['test_end'].iloc[i] - results_df['test_start'].iloc[i]).days
            weights.append(test_days)

        weights = np.array(weights) / sum(weights)

        weighted_sharpe = np.sum(results_df['test_sharpe'] * weights)
        weighted_sortino = np.sum(results_df['test_sortino'] * weights)
        weighted_annual_return = np.sum(results_df['test_annual_return'] * weights)

        # Calculate overall cumulative return (this is approximate)
        overall_cumulative = ((1 + results_df['test_cumulative_return']).cumprod().iloc[-1] - 1) \
            if len(results_df) > 0 else 0
    else:
        weighted_sharpe = np.nan
        weighted_sortino = np.nan
        weighted_annual_return = np.nan
        overall_cumulative = np.nan

    summary = {
        'results': results_df,
        'weighted_sharpe': weighted_sharpe,
        'weighted_sortino': weighted_sortino,
        'weighted_annual_return': weighted_annual_return,
        'overall_cumulative': overall_cumulative,
        'parameter_frequency': {}
    }

    # Analyze parameter frequency
    if not results_df.empty:
        for param in param_grid.keys():
            param_values = [result['best_params'][param] for result in results]
            value_counts = pd.Series(param_values).value_counts()
            summary['parameter_frequency'][param] = value_counts.to_dict()

    return summary


def run_industry_benchmark(
        tickers: List[str],
        start_date: str,
        end_date: str,
        industry_mapping: Dict[str, int],
        **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs a benchmark using industry classifications instead of clustering.

    Args:
        tickers: List of stock ticker symbols
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        industry_mapping: Dictionary mapping tickers to industry IDs
        **kwargs: Additional arguments for the backtest

    Returns:
        Tuple of (portfolio_returns, metrics)
    """
    # Create industry clusters
    industry_clusters = {}
    for ticker, industry_id in industry_mapping.items():
        if industry_id not in industry_clusters:
            industry_clusters[industry_id] = []
        industry_clusters[industry_id].append(ticker)

    # Filter for only tickers in our universe
    filtered_clusters = {}
    for industry_id, industry_tickers in industry_clusters.items():
        filtered_tickers = [t for t in industry_tickers if t in tickers]
        if filtered_tickers:  # Only include non-empty clusters
            filtered_clusters[industry_id] = filtered_tickers

    # Create a backtest instance
    backtest = StatisticalArbitrageBacktest(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        **kwargs
    )

    # Prepare data
    backtest.prepare_data()

    # Initialize portfolio returns DataFrame
    all_portfolio_returns = pd.DataFrame(index=backtest.residual_returns.index)
    all_portfolio_returns['Combined'] = 0.0

    # Initialize additional tracking variables
    current_portfolio = {}  # Tracks current holdings
    last_rebalance_date = backtest.residual_returns.index[0]
    stop_win_triggered = False

    for i, date in enumerate(backtest.residual_returns.index):
        # Check if we need to rebalance
        days_since_rebalance = (date - last_rebalance_date).days

        if i == 0 or days_since_rebalance >= backtest.rebalance_period or stop_win_triggered:
            # Update rebalance tracking
            last_rebalance_date = date
            stop_win_triggered = False

            # Get data for identifying winners/losers
            lookback_start = max(0, i - backtest.lookback_window)
            lookback_data = backtest.returns.iloc[lookback_start:i]

            # Identify winners and losers
            winners, losers = identify_winners_losers(
                lookback_data,
                filtered_clusters,
                backtest.lookback_window,
                backtest.threshold
            )

            # Construct arbitrage portfolios
            current_portfolio = construct_arbitrage_portfolios(winners, losers)

        # Calculate daily returns for current portfolio
        if current_portfolio and i < len(backtest.returns) - 1:
            # Get next day's returns
            next_day = backtest.returns.index[i + 1]
            next_day_returns = backtest.returns.loc[next_day]

            # Calculate portfolio returns
            day_returns = {}

            for industry_id, portfolio in current_portfolio.items():
                # Initialize return for this industry
                day_returns[f"Industry_{industry_id}"] = 0.0

                # Sum weighted returns
                for ticker, weight in portfolio.items():
                    if ticker in next_day_returns.index:
                        day_returns[f"Industry_{industry_id}"] += weight * next_day_returns[ticker]

            # Calculate combined return (equal weight across industries)
            if day_returns:
                day_returns['Combined'] = float(np.mean(list(day_returns.values())))
            else:
                day_returns['Combined'] = 0.0

            # Store in the all_portfolio_returns DataFrame
            for col, ret in day_returns.items():
                if col in all_portfolio_returns.columns:
                    all_portfolio_returns.loc[next_day, col] = ret
                else:
                    all_portfolio_returns[col] = 0.0
                    all_portfolio_returns.loc[next_day, col] = ret

            # Check for stop-win
            if backtest.stop_win_threshold is not None:
                # Calculate return since last rebalance
                since_rebalance = all_portfolio_returns.loc[
                    all_portfolio_returns.index > last_rebalance_date,
                    'Combined'
                ]

                # Calculate cumulative return
                cumulative_return = (1 + since_rebalance).prod() - 1

                # Check if stop-win is triggered
                if cumulative_return >= backtest.stop_win_threshold:
                    stop_win_triggered = True

    # Fill NaNs with zeros
    portfolio_returns = all_portfolio_returns.fillna(0)

    # Calculate metrics
    metrics = pd.DataFrame(index=portfolio_returns.columns)

    # Calculate metrics for each portfolio
    for col in portfolio_returns.columns:
        returns = portfolio_returns[col]

        # Annualized return
        metrics.loc[col, 'Annualized Return (%)'] = calculate_annualized_return(returns) * 100

        # Sharpe ratio
        metrics.loc[col, 'Sharpe Ratio'] = calculate_sharpe_ratio(returns)

        # Sortino ratio
        metrics.loc[col, 'Sortino Ratio'] = calculate_sortino_ratio(returns)

        # Add Maximum Drawdown calculation
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / running_max) - 1
        metrics.loc[col, 'Maximum Drawdown (%)'] = drawdown.min() * 100

        # Add Win Rate calculation
        win_rate = (returns > 0).mean()
        metrics.loc[col, 'Win Rate (%)'] = win_rate * 100

    return portfolio_returns, metrics


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from fama_french import get_fama_french_industries

    tickers = [
        'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'PG',
        'JNJ', 'WMT', 'MA', 'DIS', 'NFLX', 'ADBE', 'CRM', 'INTC', 'AMD', 'PYPL'
    ]
    start_date = '2018-01-01'
    end_date = '2022-12-31'

    # Run backtest for clustering approach
    backtest = StatisticalArbitrageBacktest(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        clustering_method='sponge_sym',
        beta_window=60,
        correlation_window=20,
        lookback_window=5,
        rebalance_period=3,
        threshold=0.0,  # Fixed: type 'float' instead of 'floating'
        stop_win_threshold=0.05,  # Fixed: type 'float' instead of 'floating'
        n_clusters_method='mp'
    )

    portfolio_returns = backtest.run_backtest()
    metrics = backtest.get_performance_metrics()

    print("Clustering Approach Metrics:")
    print(metrics)

    # Run industry benchmark
    # Here we would use Fama-French 12 industries or similar
    # For this example, we'll mock up a simple industry mapping
    industry_mapping = {
        'AAPL': 1, 'MSFT': 1, 'GOOGL': 1, 'META': 1,  # Tech
        'AMZN': 2, 'WMT': 2,  # Retail
        'JPM': 3, 'V': 3, 'MA': 3, 'PYPL': 3,  # Finance
        'PG': 4, 'JNJ': 4,  # Consumer Staples
        'TSLA': 5,  # Auto
        'NVDA': 6, 'INTC': 6, 'AMD': 6,  # Semiconductors
        'DIS': 7, 'NFLX': 7,  # Media
        'ADBE': 8, 'CRM': 8  # Software
    }

    industry_returns, industry_metrics = run_industry_benchmark(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        industry_mapping=industry_mapping,
        beta_window=60,
        correlation_window=20,
        lookback_window=5,
        rebalance_period=3,
        threshold=0.0,  # Fixed: type 'float' instead of 'floating'
        stop_win_threshold=0.05  # Fixed: type 'float' instead of 'floating'
    )

    print("\nIndustry Benchmark Metrics:")
    print(industry_metrics)

    # Compare cumulative returns
    clustering_cumulative = (1 + portfolio_returns['Combined']).cumprod() - 1
    industry_cumulative = (1 + industry_returns['Combined']).cumprod() - 1

    print("\nFinal Cumulative Returns:")
    print(f"Clustering: {clustering_cumulative.iloc[-1]:.2%}")
    print(f"Industry: {industry_cumulative.iloc[-1]:.2%}")

    # Plot comparison
    plt.figure(figsize=(12, 6))
    plt.plot(clustering_cumulative.index, clustering_cumulative, label='Clustering Approach')
    plt.plot(industry_cumulative.index, industry_cumulative, label='Industry Benchmark')
    plt.title('Cumulative Returns Comparison')
    plt.xlabel('Date')
    plt.ylabel('Return')
    plt.legend()
    plt.grid(True)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    plt.tight_layout()
    plt.show()

    # Example of backtesting extensions
    # Note: These are computationally intensive, so only run when needed

    # 1. Time series cross-validation
    print("\nRunning Time Series Cross-Validation (simplified)...")
    tscv_results = time_series_cross_validation(
        tickers[:10],  # Use fewer tickers for speed
        start_date,
        end_date,
        n_splits=3,  # Fewer splits for example
        backtest_params={
            'clustering_method': 'spectral',  # Faster method
            'lookback_window': 5,
            'rebalance_period': 3,
            'threshold': 0.0
        }
    )

    print("\nTime Series CV Results:")
    print(f"Mean Sharpe: {tscv_results['mean_sharpe']:.4f} ± {tscv_results['std_sharpe']:.4f}")
    print(f"Mean Sortino: {tscv_results['mean_sortino']:.4f} ± {tscv_results['std_sortino']:.4f}")
    print(
        f"Mean Annualized Return: {tscv_results['mean_annualized_return']:.4%} ± {tscv_results['std_annualized_return']:.4%}")

    # 2. Monte Carlo simulation (using precomputed returns for speed)
    print("\nRunning Monte Carlo Simulation (simplified)...")
    mc_results = monte_carlo_backtest(
        backtest.returns,
        backtest.residual_returns,
        n_simulations=100,  # Fewer simulations for example
        backtest_params={
            'clustering_method': 'spectral',  # Faster method
            'lookback_window': 5,
            'rebalance_period': 3,
            'threshold': 0.0
        },
        tickers=tickers[:10]  # Use fewer tickers for speed
    )

    print("\nMonte Carlo Results:")
    print(f"Mean Sharpe: {mc_results['mean']['sharpe_ratio']:.4f} ± {mc_results['std']['sharpe_ratio']:.4f}")
    print(
        f"5th-95th Percentile Sharpe: [{mc_results['5th_percentile']['sharpe_ratio']:.4f}, {mc_results['95th_percentile']['sharpe_ratio']:.4f}]")
    print(
        f"Mean Annualized Return: {mc_results['mean']['annualized_return']:.4%} ± {mc_results['std']['annualized_return']:.4%}")

    # 3. Walk-forward optimization (simplified example)
    print("\nRunning Walk-Forward Optimization (simplified)...")
    wf_results = walk_forward_optimization(
        tickers[:10],  # Use fewer tickers for speed
        start_date,
        '2020-12-31',  # Shorter period for example
        initial_window=252,
        step=63,  # Quarterly steps
        param_grid={
            'lookback_window': [3, 10],  # Simplified grid
            'rebalance_period': [1, 5],
            'threshold': [0.0, 0.005]
        },
        clustering_method='spectral'  # Faster method
    )

    print("\nWalk-Forward Optimization Results:")
    print(f"Weighted Sharpe: {wf_results['weighted_sharpe']:.4f}")
    print(f"Weighted Annualized Return: {wf_results['weighted_annual_return']:.4%}")

    print("\nParameter Frequency:")
    for param, freq in wf_results['parameter_frequency'].items():
        print(f"{param}:", end=" ")
        sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        for value, count in sorted_freq:
            print(f"{value}: {count}", end=", ")
        print()