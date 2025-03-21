import numpy as np
import pandas as pd
import scipy.optimize as sco
from typing import Dict, List, Tuple, Optional, Union
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

class PortfolioAllocator:
    """
    Manages the integration of the statistical arbitrage strategy 
    with other investment strategies for portfolio optimization.
    """
    
    def __init__(
            self,
            strategies: Dict[str, pd.DataFrame],
            risk_free_rate: float = 0.0,
            risk_budget: Optional[Dict[str, float]] = None
    ):
        """
        Initializes the portfolio allocator.
        
        Args:
            strategies: Dictionary mapping strategy names to return series
            risk_free_rate: Annual risk-free rate
            risk_budget: Optional dictionary mapping strategy names to risk budget weights
        """
        # Convert strategies to a unified DataFrame
        self.returns = pd.DataFrame()
        for name, strategy_returns in strategies.items():
            # Extract 'Combined' column if it exists (for stat arb), otherwise use the series directly
            if isinstance(strategy_returns, pd.DataFrame) and 'Combined' in strategy_returns.columns:
                self.returns[name] = strategy_returns['Combined']
            elif isinstance(strategy_returns, pd.Series):
                self.returns[name] = strategy_returns
            elif isinstance(strategy_returns, pd.DataFrame):
                if strategy_returns.shape[1] == 1:
                    self.returns[name] = strategy_returns.iloc[:, 0]
                else:
                    self.returns[name] = strategy_returns.mean(axis=1)
        
        self.strategy_names = list(self.returns.columns)
        self.risk_free_rate = risk_free_rate
        self.risk_budget = risk_budget
        
        # Calculate key metrics
        self.correlation_matrix = self.calculate_correlation_matrix()
        self.covariance_matrix = self.calculate_covariance_matrix()
        self.metrics = self.calculate_strategy_metrics()
    
    def calculate_correlation_matrix(self) -> pd.DataFrame:
        """
        Calculates correlation matrix between strategies.
        
        Returns:
            DataFrame with correlation matrix
        """
        return self.returns.corr()
    
    def calculate_covariance_matrix(self) -> pd.DataFrame:
        """
        Calculates covariance matrix between strategies.
        
        Returns:
            DataFrame with covariance matrix
        """
        # Calculate annualized covariance matrix (252 trading days)
        return self.returns.cov() * 252
    
    def calculate_strategy_metrics(self) -> pd.DataFrame:
        """
        Calculates key metrics for each strategy.
        
        Returns:
            DataFrame with strategy metrics
        """
        metrics = pd.DataFrame(index=self.strategy_names)
        
        # Calculate metrics
        for strategy in self.strategy_names:
            returns = self.returns[strategy].dropna()
            
            # Daily metrics
            mean_return = returns.mean()
            std_dev = returns.std()
            
            # Annualized metrics (252 trading days)
            ann_return = (1 + mean_return) ** 252 - 1
            ann_vol = std_dev * np.sqrt(252)
            sharpe = (ann_return - self.risk_free_rate) / ann_vol if ann_vol > 0 else 0
            
            # Drawdown metrics
            cum_returns = (1 + returns).cumprod()
            peak = cum_returns.cummax()
            drawdown = (cum_returns / peak) - 1
            max_drawdown = drawdown.min()
            
            # Correlation with other strategies
            mean_corr = self.correlation_matrix[strategy].drop(strategy).mean()
            
            # Store metrics
            metrics.loc[strategy, 'Mean Daily Return'] = mean_return
            metrics.loc[strategy, 'Daily Volatility'] = std_dev
            metrics.loc[strategy, 'Annualized Return'] = ann_return
            metrics.loc[strategy, 'Annualized Volatility'] = ann_vol
            metrics.loc[strategy, 'Sharpe Ratio'] = sharpe
            metrics.loc[strategy, 'Max Drawdown'] = max_drawdown
            metrics.loc[strategy, 'Mean Correlation'] = mean_corr
        
        return metrics
    
    def _risk_parity_weights(self) -> np.ndarray:
        """
        Calculates risk parity weights.
        
        Returns:
            Array of weights
        """
        n = len(self.strategy_names)
        init_weights = np.ones(n) / n
        bounds = tuple((0.001, 1) for _ in range(n))
        
        # Risk parity objective function
        def objective(weights):
            # Normalize weights
            weights = weights / np.sum(weights)
            
            # Calculate portfolio risk
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.covariance_matrix.values, weights)))
            
            # Calculate risk contribution
            risk_contribution = weights * np.dot(self.covariance_matrix.values, weights) / portfolio_vol
            
            # Target equal risk contribution
            target_risk = portfolio_vol / n
            risk_diffs = risk_contribution - target_risk
            
            return np.sum(risk_diffs ** 2)
        
        # Constraint: weights sum to 1
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        
        # Optimize
        result = sco.minimize(
            objective, init_weights, method='SLSQP',
            bounds=bounds, constraints=constraints,
            options={'disp': False}
        )
        
        if result.success:
            return result.x
        else:
            # Fallback to equal weights
            return np.ones(n) / n
    
    def _minimum_variance_weights(self) -> np.ndarray:
        """
        Calculates minimum variance portfolio weights.
        
        Returns:
            Array of weights
        """
        n = len(self.strategy_names)
        init_weights = np.ones(n) / n
        bounds = tuple((0.001, 1) for _ in range(n))
        
        # Minimum variance objective function
        def objective(weights):
            return np.dot(weights.T, np.dot(self.covariance_matrix.values, weights))
        
        # Constraint: weights sum to 1
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        
        # Optimize
        result = sco.minimize(
            objective, init_weights, method='SLSQP',
            bounds=bounds, constraints=constraints,
            options={'disp': False}
        )
        
        if result.success:
            return result.x
        else:
            # Fallback to equal weights
            return np.ones(n) / n
    
    def _maximum_diversification_weights(self) -> np.ndarray:
        """
        Calculates maximum diversification portfolio weights.
        
        Returns:
            Array of weights
        """
        n = len(self.strategy_names)
        init_weights = np.ones(n) / n
        bounds = tuple((0.001, 1) for _ in range(n))
        
        # Maximum diversification objective function (minimize concentration)
        def objective(weights):
            # Normalize weights
            weights = weights / np.sum(weights)
            
            # Calculate portfolio volatility
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.covariance_matrix.values, weights)))
            
            # Calculate weighted average of individual volatilities
            weighted_vols = np.sum(weights * np.sqrt(np.diag(self.covariance_matrix.values)))
            
            # Diversification ratio (higher is better, so we minimize negative)
            div_ratio = weighted_vols / portfolio_vol
            
            return -div_ratio
        
        # Constraint: weights sum to 1
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        
        # Optimize
        result = sco.minimize(
            objective, init_weights, method='SLSQP',
            bounds=bounds, constraints=constraints,
            options={'disp': False}
        )
        
        if result.success:
            return result.x
        else:
            # Fallback to equal weights
            return np.ones(n) / n
    
    def _risk_budget_weights(self) -> np.ndarray:
        """
        Calculates weights based on risk budget.
        
        Returns:
            Array of weights
        """
        n = len(self.strategy_names)
        
        if self.risk_budget is None:
            # Equal risk budget
            risk_budget = np.ones(n) / n
        else:
            # Use specified risk budget
            risk_budget = np.array([self.risk_budget.get(s, 1/n) for s in self.strategy_names])
            # Normalize to sum to 1
            risk_budget = risk_budget / np.sum(risk_budget)
        
        init_weights = np.ones(n) / n
        bounds = tuple((0.001, 1) for _ in range(n))
        
        # Risk budgeting objective function
        def objective(weights):
            # Normalize weights
            weights = weights / np.sum(weights)
            
            # Calculate portfolio risk
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.covariance_matrix.values, weights)))
            
            # Calculate risk contribution
            risk_contribution = weights * np.dot(self.covariance_matrix.values, weights) / portfolio_vol
            
            # Normalize risk contribution
            risk_contribution_proportion = risk_contribution / np.sum(risk_contribution)
            
            # Compare to target risk budget
            risk_diffs = risk_contribution_proportion - risk_budget
            
            return np.sum(risk_diffs ** 2)
        
        # Constraint: weights sum to 1
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        
        # Optimize
        result = sco.minimize(
            objective, init_weights, method='SLSQP',
            bounds=bounds, constraints=constraints,
            options={'disp': False}
        )
        
        if result.success:
            return result.x
        else:
            # Fallback to proportional to risk budget
            return risk_budget
    
    def optimize_allocation(self, method: str = 'risk_parity') -> pd.Series:
        """
        Optimizes portfolio allocation.
        
        Args:
            method: Allocation method ('risk_parity', 'min_variance', 'max_diversification', 'risk_budget')
            
        Returns:
            Series with optimal weights
        """
        if method == 'risk_parity':
            weights = self._risk_parity_weights()
        elif method == 'min_variance':
            weights = self._minimum_variance_weights()
        elif method == 'max_diversification':
            weights = self._maximum_diversification_weights()
        elif method == 'risk_budget':
            weights = self._risk_budget_weights()
        elif method == 'equal_weight':
            weights = np.ones(len(self.strategy_names)) / len(self.strategy_names)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Convert to Series
        return pd.Series(weights, index=self.strategy_names)
    
    def calculate_portfolio_metrics(self, weights: pd.Series) -> Dict:
        """
        Calculates metrics for a portfolio with the given weights.
        
        Args:
            weights: Series with strategy weights
            
        Returns:
            Dictionary with portfolio metrics
        """
        # Normalize weights
        weights = weights / weights.sum()
        
        # Calculate weighted returns
        portfolio_returns = self.returns.dot(weights)
        
        # Portfolio metrics
        mean_return = portfolio_returns.mean()
        std_dev = portfolio_returns.std()
        
        # Annualized metrics (252 trading days)
        ann_return = (1 + mean_return) ** 252 - 1
        ann_vol = std_dev * np.sqrt(252)
        sharpe = (ann_return - self.risk_free_rate) / ann_vol if ann_vol > 0 else 0
        
        # Drawdown metrics
        cum_returns = (1 + portfolio_returns).cumprod()
        peak = cum_returns.cummax()
        drawdown = (cum_returns / peak) - 1
        max_drawdown = drawdown.min()
        
        # Calculate diversification ratio
        strategy_vols = np.sqrt(np.diag(self.covariance_matrix.values))
        weighted_vols = np.sum(weights.values * strategy_vols)
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.covariance_matrix.values, weights)))
        div_ratio = weighted_vols / portfolio_vol
        
        return {
            'Mean Daily Return': mean_return,
            'Daily Volatility': std_dev,
            'Annualized Return': ann_return,
            'Annualized Volatility': ann_vol,
            'Sharpe Ratio': sharpe,
            'Max Drawdown': max_drawdown,
            'Diversification Ratio': div_ratio
        }
    
    def calculate_information_ratio_impact(
            self,
            existing_portfolio: pd.Series,
            new_strategy: str,
            allocation_range: List[float] = [0, 0.05, 0.1, 0.2, 0.3]
    ) -> pd.DataFrame:
        """
        Calculates the impact of adding a new strategy to an existing portfolio.
        
        Args:
            existing_portfolio: Series with existing portfolio returns
            new_strategy: Name of the new strategy to add
            allocation_range: List of allocation percentages to test
            
        Returns:
            DataFrame with impact metrics at different allocation levels
        """
        if new_strategy not in self.returns.columns:
            raise ValueError(f"Strategy {new_strategy} not found")
        
        new_strategy_returns = self.returns[new_strategy]
        
        # Calculate metrics at different allocation levels
        results = []
        
        for allocation in allocation_range:
            # Combine returns
            combined_returns = (1 - allocation) * existing_portfolio + allocation * new_strategy_returns
            
            # Calculate metrics
            mean_return = combined_returns.mean()
            std_dev = combined_returns.std()
            ann_return = (1 + mean_return) ** 252 - 1
            ann_vol = std_dev * np.sqrt(252)
            sharpe = (ann_return - self.risk_free_rate) / ann_vol if ann_vol > 0 else 0
            
            # Calculate tracking error
            tracking_error = (combined_returns - existing_portfolio).std() * np.sqrt(252)
            
            # Calculate information ratio
            if tracking_error > 0:
                excess_return = ann_return - (1 + existing_portfolio.mean()) ** 252 + 1
                info_ratio = excess_return / tracking_error
            else:
                info_ratio = 0
            
            results.append({
                'Allocation': allocation,
                'Annualized Return': ann_return,
                'Annualized Volatility': ann_vol,
                'Sharpe Ratio': sharpe,
                'Tracking Error': tracking_error,
                'Information Ratio': info_ratio
            })
        
        return pd.DataFrame(results)

    def cluster_strategies(self, n_clusters: Optional[int] = None, threshold: float = 0.5) -> Dict:
        """
        Groups strategies by their correlation characteristics with improved compatibility.

        Args:
            n_clusters: Number of clusters (or None for automatic)
            threshold: Distance threshold for clustering

        Returns:
            Dictionary with clustering results
        """
        # Convert correlation to distance
        distance_matrix = 1 - np.abs(self.correlation_matrix.values)

        # Apply hierarchical clustering with improved compatibility for different sklearn versions
        try:
            # Try the standard approach first
            if n_clusters is not None:
                clustering = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    affinity='precomputed',
                    linkage='complete'
                )
            else:
                clustering = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=threshold,
                    affinity='precomputed',
                    linkage='complete'
                )

            labels = clustering.fit_predict(distance_matrix)

        except TypeError as e:
            print(f"Warning: {e}")
            print("Trying alternative clustering approach...")

            # For older scikit-learn versions that don't support 'affinity' with distance_threshold
            if n_clusters is not None:
                # Use n_clusters approach
                from scipy.cluster.hierarchy import linkage, fcluster

                # Convert distance matrix to condensed form
                from scipy.spatial.distance import squareform
                condensed_dist = squareform(distance_matrix)

                # Perform hierarchical clustering
                Z = linkage(condensed_dist, method='complete')

                # Cut the dendrogram to get n_clusters
                labels = fcluster(Z, n_clusters, criterion='maxclust') - 1  # zero-based

            else:
                # Use standard approach with default affinity
                # This may work for some versions but input data needs to be features
                # For those cases, we'll use a fixed number of clusters as fallback
                print("Using fixed number of clusters as fallback")
                n_clusters = max(2, len(self.strategy_names) // 3)  # Reasonable default

                clustering = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    linkage='complete'
                )

                # Use correlation as features (not ideal but will work)
                labels = clustering.fit_predict(self.correlation_matrix.values)

        # Group strategies by cluster
        clusters = {}
        for i, strategy in enumerate(self.strategy_names):
            cluster_id = labels[i]
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(strategy)

        return {
            'labels': labels,
            'clusters': clusters,
            'n_clusters': len(clusters)
        }
    
    def plot_correlation_matrix(self, **kwargs):
        """
        Plots the correlation matrix between strategies.
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot heatmap
        im = ax.imshow(self.correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Correlation')
        
        # Add labels
        ax.set_xticks(np.arange(len(self.strategy_names)))
        ax.set_yticks(np.arange(len(self.strategy_names)))
        ax.set_xticklabels(self.strategy_names, rotation=45, ha='right')
        ax.set_yticklabels(self.strategy_names)
        
        # Add title
        ax.set_title('Strategy Correlation Matrix')
        
        # Add grid
        for i in range(len(self.strategy_names)):
            for j in range(len(self.strategy_names)):
                ax.text(j, i, f"{self.correlation_matrix.iloc[i, j]:.2f}",
                       ha="center", va="center", color="w" if abs(self.correlation_matrix.iloc[i, j]) > 0.5 else "black")
        
        plt.tight_layout()
        plt.show()
    
    def plot_efficient_frontier(self, n_portfolios: int = 1000, **kwargs):
        """
        Plots the efficient frontier with strategy positions.
        
        Args:
            n_portfolios: Number of random portfolios to generate
        """
        n = len(self.strategy_names)
        
        # Generate random portfolios
        np.random.seed(42)
        weights = np.random.random((n_portfolios, n))
        weights = weights / np.sum(weights, axis=1).reshape(-1, 1)
        
        # Calculate returns and volatilities
        portfolio_returns = np.zeros(n_portfolios)
        portfolio_volatilities = np.zeros(n_portfolios)
        
        for i in range(n_portfolios):
            portfolio_returns[i] = np.sum(weights[i] * self.metrics['Mean Daily Return'].values) * 252
            portfolio_volatilities[i] = np.sqrt(np.dot(weights[i].T, np.dot(
                self.covariance_matrix.values, weights[i])))
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot random portfolios
        scatter = ax.scatter(portfolio_volatilities, portfolio_returns, 
                   c=portfolio_returns / portfolio_volatilities, marker='o', 
                   cmap='viridis', alpha=0.5)
        
        # Plot individual strategies
        for i, strategy in enumerate(self.strategy_names):
            ax.scatter(
                self.metrics.loc[strategy, 'Annualized Volatility'],
                self.metrics.loc[strategy, 'Annualized Return'],
                marker='*', s=200, label=strategy
            )
        
        # Plot optimized portfolios
        methods = ['risk_parity', 'min_variance', 'max_diversification']
        markers = ['o', 's', 'X']
        
        for method, marker in zip(methods, markers):
            weights = self.optimize_allocation(method)
            metrics = self.calculate_portfolio_metrics(weights)
            
            ax.scatter(
                metrics['Annualized Volatility'],
                metrics['Annualized Return'],
                marker=marker, s=200, 
                label=f"{method.replace('_', ' ').title()}"
            )
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, label='Sharpe Ratio')
        
        # Add labels and legend
        ax.set_xlabel('Annualized Volatility')
        ax.set_ylabel('Annualized Return')
        ax.set_title('Efficient Frontier')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def get_summary_report(self) -> pd.DataFrame:
        """
        Generates a summary report of all strategies and optimized portfolios.
        
        Returns:
            DataFrame with summary metrics
        """
        # Start with strategy metrics
        summary = self.metrics[['Annualized Return', 'Annualized Volatility', 
                              'Sharpe Ratio', 'Max Drawdown']].copy()
        
        # Add optimized portfolios
        methods = ['risk_parity', 'min_variance', 'max_diversification', 'equal_weight']
        
        for method in methods:
            weights = self.optimize_allocation(method)
            metrics = self.calculate_portfolio_metrics(weights)
            
            # Add to summary
            method_name = f"Portfolio ({method.replace('_', ' ').title()})"
            summary.loc[method_name] = [
                metrics['Annualized Return'],
                metrics['Annualized Volatility'],
                metrics['Sharpe Ratio'],
                metrics['Max Drawdown']
            ]
        
        return summary

def calculate_information_ratio_impact(
    existing_portfolio_returns: pd.Series,
    stat_arb_returns: pd.Series,
    allocation: float = 0.1
) -> Dict:
    """
    Calculates the impact of adding statistical arbitrage to an existing portfolio.
    
    Args:
        existing_portfolio_returns: Series with existing portfolio returns
        stat_arb_returns: Series with statistical arbitrage returns
        allocation: Allocation to statistical arbitrage
        
    Returns:
        Dictionary with impact metrics
    """
    # Align dates
    common_dates = existing_portfolio_returns.index.intersection(stat_arb_returns.index)
    existing = existing_portfolio_returns[common_dates]
    stat_arb = stat_arb_returns[common_dates]
    
    # Calculate combined returns
    combined_returns = existing * (1 - allocation) + stat_arb * allocation
    
    # Calculate metrics
    existing_mean = existing.mean() * 252
    existing_std = existing.std() * np.sqrt(252)
    existing_sharpe = existing_mean / existing_std
    
    combined_mean = combined_returns.mean() * 252
    combined_std = combined_returns.std() * np.sqrt(252)
    combined_sharpe = combined_mean / combined_std
    
    # Calculate tracking error
    tracking_error = (combined_returns - existing).std() * np.sqrt(252)
    
    # Calculate information ratio
    excess_return = combined_mean - existing_mean
    info_ratio = excess_return / tracking_error if tracking_error > 0 else 0
    
    return {
        'base_return': existing_mean,
        'combined_return': combined_mean,
        'return_improvement': combined_mean - existing_mean,
        'return_improvement_pct': (combined_mean - existing_mean) / abs(existing_mean) * 100,
        
        'base_volatility': existing_std,
        'combined_volatility': combined_std,
        'volatility_change': combined_std - existing_std,
        'volatility_change_pct': (combined_std - existing_std) / existing_std * 100,
        
        'base_sharpe': existing_sharpe,
        'combined_sharpe': combined_sharpe,
        'sharpe_improvement': combined_sharpe - existing_sharpe,
        'sharpe_improvement_pct': (combined_sharpe - existing_sharpe) / existing_sharpe * 100,
        
        'tracking_error': tracking_error,
        'information_ratio': info_ratio
    }

if __name__ == "__main__":
    # Example usage
    import numpy as np
    import pandas as pd
    from datetime import datetime, timedelta
    
    # Create sample returns
    np.random.seed(42)
    dates = pd.date_range(start='2018-01-01', end='2022-12-31', freq='B')
    
    # Create mock strategy returns
    strategies = {}
    
    # S&P 500 market index (higher return, higher volatility)
    strategies['SPY'] = pd.Series(
        np.random.normal(0.0005, 0.012, len(dates)),
        index=dates
    )
    
    # Trend following strategy (lower return, lower volatility, low correlation)
    strategies['Trend'] = pd.Series(
        np.random.normal(0.0003, 0.009, len(dates)),
        index=dates
    )
    
    # Value strategy (similar return, different correlation pattern)
    strategies['Value'] = pd.Series(
        np.random.normal(0.0004, 0.011, len(dates)),
        index=dates
    )
    
    # Statistical arbitrage (low correlation, low volatility)
    stat_arb_returns = np.random.normal(0.0003, 0.006, len(dates))
    # Add mean-reversion property
    for i in range(1, len(stat_arb_returns)):
        if stat_arb_returns[i-1] < -0.01:
            stat_arb_returns[i] += 0.003
        elif stat_arb_returns[i-1] > 0.01:
            stat_arb_returns[i] -= 0.003
    
    strategies['StatArb'] = pd.Series(stat_arb_returns, index=dates)
    
    # Fixed income (lowest volatility, negative correlation with equity)
    fi_returns = np.random.normal(0.0002, 0.003, len(dates))
    for i in range(len(fi_returns)):
        fi_returns[i] = fi_returns[i] - 0.3 * strategies['SPY'].iloc[i]
    
    strategies['FixedIncome'] = pd.Series(fi_returns, index=dates)
    
    # Initialize portfolio allocator
    allocator = PortfolioAllocator(strategies, risk_free_rate=0.005)
    
    # Print strategy metrics
    print("Strategy Metrics:")
    print(allocator.metrics)
    
    # Calculate correlations
    print("\nCorrelation Matrix:")
    print(allocator.correlation_matrix)
    
    # Optimize allocation
    for method in ['risk_parity', 'min_variance', 'max_diversification', 'equal_weight']:
        weights = allocator.optimize_allocation(method)
        print(f"\n{method.replace('_', ' ').title()} Weights:")
        print(weights)
        
        metrics = allocator.calculate_portfolio_metrics(weights)
        print(f"\n{method.replace('_', ' ').title()} Portfolio Metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
    
    # Cluster strategies
    clustering = allocator.cluster_strategies()
    print("\nStrategy Clusters:")
    for cluster_id, strategies in clustering['clusters'].items():
        print(f"Cluster {cluster_id}: {', '.join(strategies)}")
    
    # Calculate impact of adding statistical arbitrage to SPY
    impact = calculate_information_ratio_impact(
        strategies['SPY'],
        strategies['StatArb'],
        allocation=0.15
    )
    
    print("\nImpact of Adding Statistical Arbitrage to SPY:")
    for key, value in impact.items():
        print(f"{key}: {value:.4f}")
