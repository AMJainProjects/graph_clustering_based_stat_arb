import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

def calculate_cluster_mean_returns(
    returns: pd.DataFrame,
    clusters: Dict[int, List[str]],
    lookback_window: int = 5
) -> Dict[int, float]:
    """
    Calculates the mean return of each cluster over the lookback window.
    
    Args:
        returns: DataFrame with daily returns
        clusters: Dictionary mapping cluster IDs to lists of tickers
        lookback_window: Number of days to look back for calculating mean returns
        
    Returns:
        Dictionary mapping cluster IDs to mean returns
    """
    cluster_means = {}
    
    # Use the last lookback_window days of data
    recent_returns = returns.iloc[-lookback_window:]
    
    for cluster_id, tickers in clusters.items():
        # Select only the tickers in this cluster
        cluster_returns = recent_returns[tickers]
        
        # Calculate the mean return across all stocks in the cluster for each day
        daily_means = cluster_returns.mean(axis=1)
        
        # Calculate the overall mean return for the cluster
        cluster_means[cluster_id] = daily_means.mean()
    
    return cluster_means

def identify_winners_losers(
    returns: pd.DataFrame,
    clusters: Dict[int, List[str]],
    lookback_window: int = 5,
    threshold: float = 0.0
) -> Tuple[Dict[int, List[str]], Dict[int, List[str]]]:
    """
    Identifies winners and losers in each cluster based on cumulative deviation from cluster mean.
    
    Args:
        returns: DataFrame with daily returns
        clusters: Dictionary mapping cluster IDs to lists of tickers
        lookback_window: Number of days to look back
        threshold: Threshold for classifying as winner/loser
        
    Returns:
        Tuple of (winners, losers) dictionaries mapping cluster IDs to lists of tickers
    """
    winners = {cluster_id: [] for cluster_id in clusters}
    losers = {cluster_id: [] for cluster_id in clusters}
    
    # Use the last lookback_window days of data
    recent_returns = returns.iloc[-lookback_window:]
    
    for cluster_id, tickers in clusters.items():
        # Select only the tickers in this cluster
        cluster_returns = recent_returns[tickers]
        
        # Calculate daily mean returns for the cluster
        daily_cluster_means = cluster_returns.mean(axis=1)
        
        # Calculate cumulative deviation from cluster mean for each stock
        cumulative_deviations = {}
        
        for ticker in tickers:
            stock_returns = cluster_returns[ticker]
            daily_deviations = stock_returns - daily_cluster_means
            cumulative_deviation = daily_deviations.sum()
            cumulative_deviations[ticker] = cumulative_deviation
        
        # Classify as winners or losers based on threshold
        for ticker, deviation in cumulative_deviations.items():
            if deviation > threshold:
                winners[cluster_id].append(ticker)
            elif deviation < -threshold:
                losers[cluster_id].append(ticker)
    
    return winners, losers

def construct_arbitrage_portfolios(
    winners: Dict[int, List[str]],
    losers: Dict[int, List[str]]
) -> Dict[int, Dict[str, float]]:
    """
    Constructs zero-cost arbitrage portfolios within each cluster.
    
    Args:
        winners: Dictionary mapping cluster IDs to lists of winner tickers
        losers: Dictionary mapping cluster IDs to lists of loser tickers
        
    Returns:
        Dictionary mapping cluster IDs to portfolios (ticker -> weight mappings)
    """
    portfolios = {}
    
    for cluster_id in winners.keys():
        cluster_winners = winners[cluster_id]
        cluster_losers = losers[cluster_id]
        
        # Skip clusters where we can't form a zero-cost portfolio
        if not cluster_winners or not cluster_losers:
            continue
        
        # Calculate weights for long (losers) and short (winners) positions
        # Equal dollar amounts on each side (zero-cost)
        short_weight = -1.0 / len(cluster_winners) if cluster_winners else 0
        long_weight = 1.0 / len(cluster_losers) if cluster_losers else 0
        
        # Create the portfolio for this cluster
        portfolio = {}
        
        # Short the winners
        for ticker in cluster_winners:
            portfolio[ticker] = short_weight
        
        # Long the losers
        for ticker in cluster_losers:
            portfolio[ticker] = long_weight
        
        portfolios[cluster_id] = portfolio
    
    return portfolios

def calculate_portfolio_returns(
    returns: pd.DataFrame,
    portfolios: Dict[int, Dict[str, float]],
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    previous_portfolios: Optional[Dict[int, Dict[str, float]]] = None,
    transaction_cost: float = 0.0005  # 5 basis points (0.05%)
) -> pd.DataFrame:
    """
    Calculates the returns of the arbitrage portfolios including transaction costs.
    
    Args:
        returns: DataFrame with daily returns
        portfolios: Dictionary mapping cluster IDs to portfolios
        start_date: Optional start date for calculating returns
        end_date: Optional end date for calculating returns
        previous_portfolios: Previous portfolio positions for calculating turnover
        transaction_cost: Transaction cost as a decimal (e.g., 0.0005 for 5 bps)
        
    Returns:
        DataFrame with daily returns for each cluster portfolio
    """
    # Filter returns by date range if specified
    if start_date is not None or end_date is not None:
        date_mask = pd.Series(True, index=returns.index)
        
        if start_date is not None:
            date_mask &= (returns.index >= start_date)
        
        if end_date is not None:
            date_mask &= (returns.index <= end_date)
        
        returns = returns.loc[date_mask]
    
    # Initialize DataFrame for portfolio returns
    portfolio_returns = pd.DataFrame(
        index=returns.index,
        columns=[f"Cluster_{cluster_id}" for cluster_id in portfolios.keys()]
    )
    
    # Add columns for transaction costs
    if previous_portfolios is not None:
        portfolio_returns[[f"Cluster_{cluster_id}_TC" for cluster_id in portfolios.keys()]] = 0.0
        portfolio_returns['Combined_TC'] = 0.0
    
    # Calculate transaction costs if previous portfolios are provided
    transaction_costs = {}
    if previous_portfolios is not None:
        for cluster_id, portfolio in portfolios.items():
            # Get previous portfolio for this cluster
            prev_portfolio = previous_portfolios.get(cluster_id, {})
            
            # Calculate turnover (sum of absolute weight changes)
            turnover = 0.0
            all_tickers = set(portfolio.keys()).union(set(prev_portfolio.keys()))
            
            for ticker in all_tickers:
                current_weight = portfolio.get(ticker, 0.0)
                previous_weight = prev_portfolio.get(ticker, 0.0)
                turnover += abs(current_weight - previous_weight)
            
            # Calculate transaction cost for this cluster (applied on day 1)
            transaction_costs[cluster_id] = turnover * transaction_cost
    
    # Calculate returns for each cluster portfolio
    for cluster_id, portfolio in portfolios.items():
        # Extract the tickers and weights
        tickers = list(portfolio.keys())
        weights = np.array(list(portfolio.values()))
        
        # Calculate weighted returns for each day
        cluster_returns = []
        
        for day_idx, day in enumerate(returns.index):
            # Get returns for this day
            day_returns = returns.loc[day, tickers].values
            weighted_return = day_returns @ weights
            
            # Apply transaction costs only on the first day
            if day_idx == 0 and cluster_id in transaction_costs:
                tc = transaction_costs[cluster_id]
                weighted_return -= tc
                
                # Store transaction cost separately if tracking
                if previous_portfolios is not None:
                    portfolio_returns.loc[day, f"Cluster_{cluster_id}_TC"] = tc
            
            cluster_returns.append(weighted_return)
        
        # Store in the DataFrame
        portfolio_returns[f"Cluster_{cluster_id}"] = cluster_returns
    
    # Add a combined portfolio (equal weight across all clusters)
    if len(portfolios) > 0:
        portfolio_returns['Combined'] = portfolio_returns[[f"Cluster_{cluster_id}" for cluster_id in portfolios.keys()]].mean(axis=1)
        
        # Calculate combined transaction costs if tracking
        if previous_portfolios is not None:
            portfolio_returns['Combined_TC'] = portfolio_returns[[f"Cluster_{cluster_id}_TC" for cluster_id in portfolios.keys()]].mean(axis=1)
    
    return portfolio_returns

def check_stop_win(
    portfolio_returns: pd.DataFrame,
    threshold: float = 0.05
) -> bool:
    """
    Checks if the cumulative return has reached the stop-win threshold.
    
    Args:
        portfolio_returns: DataFrame with daily portfolio returns
        threshold: Stop-win threshold as a decimal (e.g., 0.05 for 5%)
        
    Returns:
        True if threshold is reached, False otherwise
    """
    # Calculate cumulative return
    cumulative_return = (1 + portfolio_returns['Combined']).prod() - 1
    
    # Check if it exceeds the threshold
    return cumulative_return >= threshold

def construct_and_evaluate_portfolios(
    returns: pd.DataFrame,
    clusters: Dict[int, List[str]],
    lookback_window: int = 5,
    threshold: float = 0.0,
    holding_period: int = 3,
    stop_win_threshold: Optional[float] = None
) -> Tuple[Dict[int, Dict[str, float]], pd.DataFrame]:
    """
    Constructs and evaluates arbitrage portfolios based on clusters.
    
    Args:
        returns: DataFrame with daily returns
        clusters: Dictionary mapping cluster IDs to lists of tickers
        lookback_window: Number of days to look back for identifying winners/losers
        threshold: Threshold for classifying as winner/loser
        holding_period: Number of days to hold the portfolio
        stop_win_threshold: Optional stop-win threshold
        
    Returns:
        Tuple of (portfolios, portfolio_returns)
    """
    # Identify winners and losers
    winners, losers = identify_winners_losers(
        returns, clusters, lookback_window, threshold
    )
    
    # Construct arbitrage portfolios
    portfolios = construct_arbitrage_portfolios(winners, losers)
    
    # Calculate future returns for the holding period
    if holding_period > 0 and holding_period <= len(returns):
        future_returns = returns.iloc[-holding_period:]
        
        # Calculate portfolio returns
        portfolio_returns = calculate_portfolio_returns(
            future_returns, portfolios
        )
        
        # Check for stop-win if threshold is provided
        if stop_win_threshold is not None:
            # Calculate daily cumulative returns
            cumulative_returns = (1 + portfolio_returns).cumprod() - 1
            
            # Check each day if stop-win is triggered
            for i in range(len(cumulative_returns)):
                if cumulative_returns['Combined'].iloc[i] >= stop_win_threshold:
                    # Stop-win triggered, truncate returns
                    portfolio_returns = portfolio_returns.iloc[:i+1]
                    break
    else:
        # No holding period specified, return empty DataFrame
        portfolio_returns = pd.DataFrame(columns=['Combined'])
    
    return portfolios, portfolio_returns

if __name__ == "__main__":
    # Example usage
    from data_preprocessing import prepare_stock_data
    from correlation_matrix import compute_correlation_matrix
    from clustering import cluster_stocks, get_clusters_dict
    
    tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'PG']
    start_date = '2020-01-01'
    end_date = '2022-12-31'
    
    returns, betas, residual_returns = prepare_stock_data(tickers, start_date, end_date)
    
    # Compute correlation matrix
    corr_matrix = compute_correlation_matrix(residual_returns)
    
    # Cluster stocks
    labels, n_clusters = cluster_stocks(
        corr_matrix, method='sponge_sym', lookback_periods=60
    )
    
    clusters = get_clusters_dict(labels, residual_returns.columns.tolist())
    
    # Construct and evaluate portfolios
    portfolios, portfolio_returns = construct_and_evaluate_portfolios(
        returns, clusters, lookback_window=5, threshold=0.0,
        holding_period=3, stop_win_threshold=0.05
    )
    
    # Print portfolio compositions
    print("Portfolio Compositions:")
    for cluster_id, portfolio in portfolios.items():
        print(f"\nCluster {cluster_id}:")
        for ticker, weight in portfolio.items():
            position = "LONG" if weight > 0 else "SHORT"
            print(f"  {ticker}: {position} {abs(weight):.4f}")
    
    # Print portfolio returns
    print("\nPortfolio Returns:")
    print(portfolio_returns)
    
    # Calculate cumulative returns
    cumulative_returns = (1 + portfolio_returns).cumprod() - 1
    print("\nCumulative Returns:")
    print(cumulative_returns.iloc[-1])
