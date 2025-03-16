import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score

def calculate_annualized_return(returns: pd.Series) -> float:
    """
    Calculates the annualized return from a series of returns.
    
    Args:
        returns: Series of returns
        
    Returns:
        Annualized return
    """
    # Count number of trading days
    n_days = len(returns)
    
    # Count number of years (assuming ~252 trading days per year)
    n_years = n_days / 252
    
    # Calculate cumulative return
    cumulative_return = (1 + returns).prod() - 1
    
    # Calculate annualized return
    if n_years > 0:
        annualized_return = (1 + cumulative_return) ** (1 / n_years) - 1
    else:
        annualized_return = 0.0
    
    return annualized_return

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Calculates the Sharpe ratio from a series of returns.
    
    Args:
        returns: Series of returns
        risk_free_rate: Risk-free rate (annual)
        
    Returns:
        Sharpe ratio
    """
    # Convert annual risk-free rate to daily
    daily_risk_free = (1 + risk_free_rate) ** (1/252) - 1
    
    # Calculate excess returns
    excess_returns = returns - daily_risk_free
    
    # Calculate Sharpe ratio
    if len(excess_returns) > 1:
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
    else:
        sharpe_ratio = 0.0
    
    return sharpe_ratio

def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Calculates the Sortino ratio from a series of returns.
    
    Args:
        returns: Series of returns
        risk_free_rate: Risk-free rate (annual)
        
    Returns:
        Sortino ratio
    """
    # Convert annual risk-free rate to daily
    daily_risk_free = (1 + risk_free_rate) ** (1/252) - 1
    
    # Calculate excess returns
    excess_returns = returns - daily_risk_free
    
    # Calculate downside deviation (only negative returns)
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) > 1 and downside_returns.std() > 0:
        downside_deviation = downside_returns.std() * np.sqrt(252)
        sortino_ratio = excess_returns.mean() * 252 / downside_deviation
    else:
        # If no downside returns or not enough data
        sortino_ratio = 0.0
    
    return sortino_ratio

def plot_correlation_matrix(correlation_matrix: pd.DataFrame, title: str = 'Correlation Matrix'):
    """
    Plots a heatmap of the correlation matrix.
    
    Args:
        correlation_matrix: DataFrame with correlation matrix
        title: Plot title
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='Correlation')
    plt.title(title)
    
    # Add ticker labels
    plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
    plt.yticks(range(len(correlation_matrix.index)), correlation_matrix.index)
    
    plt.tight_layout()
    plt.show()

def plot_clusters(
    correlation_matrix: pd.DataFrame,
    labels: np.ndarray,
    ticker_names: Optional[List[str]] = None
):
    """
    Plots the correlation matrix with stocks sorted by cluster.
    
    Args:
        correlation_matrix: DataFrame with correlation matrix
        labels: Array of cluster assignments
        ticker_names: Optional list of ticker names (if not in correlation_matrix)
    """
    if ticker_names is None:
        ticker_names = correlation_matrix.columns.tolist()
    
    # Create a mapping from tickers to indices
    ticker_indices = {ticker: i for i, ticker in enumerate(ticker_names)}
    
    # Sort tickers by cluster
    sorted_indices = np.argsort(labels)
    sorted_tickers = [ticker_names[i] for i in sorted_indices]
    
    # Reorder the correlation matrix
    sorted_corr = correlation_matrix.loc[sorted_tickers, sorted_tickers]
    
    # Plot the sorted correlation matrix
    plt.figure(figsize=(12, 10))
    plt.imshow(sorted_corr, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='Correlation')
    plt.title('Correlation Matrix Sorted by Clusters')
    
    # Add ticker labels
    plt.xticks(range(len(sorted_tickers)), sorted_tickers, rotation=90)
    plt.yticks(range(len(sorted_tickers)), sorted_tickers)
    
    # Add cluster boundaries
    cluster_boundaries = []
    current_cluster = labels[sorted_indices[0]]
    
    for i, idx in enumerate(sorted_indices[1:], 1):
        if labels[idx] != current_cluster:
            cluster_boundaries.append(i - 0.5)
            current_cluster = labels[idx]
    
    for boundary in cluster_boundaries:
        plt.axhline(y=boundary, color='black', linestyle='-', linewidth=1)
        plt.axvline(x=boundary, color='black', linestyle='-', linewidth=1)
    
    plt.tight_layout()
    plt.show()

def calculate_cluster_similarity(labels1: np.ndarray, labels2: np.ndarray) -> float:
    """
    Calculates the similarity between two clusterings using Adjusted Rand Index.
    
    Args:
        labels1: First set of cluster labels
        labels2: Second set of cluster labels
        
    Returns:
        Adjusted Rand Index (ARI)
    """
    return adjusted_rand_score(labels1, labels2)

def get_eigenvalue_distribution(correlation_matrix: pd.DataFrame):
    """
    Plots the eigenvalue distribution of the correlation matrix.
    
    Args:
        correlation_matrix: DataFrame with correlation matrix
    """
    # Calculate eigenvalues
    eigenvalues = np.linalg.eigvalsh(correlation_matrix)
    
    # Sort in descending order
    eigenvalues = np.sort(eigenvalues)[::-1]
    
    # Calculate cumulative variance explained
    total_variance = np.sum(eigenvalues)
    cumulative_variance_ratio = np.cumsum(eigenvalues) / total_variance
    
    # Plot eigenvalues
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, 'o-')
    plt.title('Eigenvalue Distribution')
    plt.xlabel('Eigenvalue Number')
    plt.ylabel('Eigenvalue')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(eigenvalues) + 1), cumulative_variance_ratio, 'o-')
    plt.title('Cumulative Variance Explained')
    plt.xlabel('Number of Eigenvalues')
    plt.ylabel('Cumulative Variance Ratio')
    plt.axhline(y=0.9, color='r', linestyle='--', label='90%')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return eigenvalues, cumulative_variance_ratio

def convert_to_annual_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Converts daily metrics to annual metrics.
    
    Args:
        metrics: DataFrame with daily metrics
        
    Returns:
        DataFrame with annual metrics
    """
    annual_metrics = metrics.copy()
    
    # Convert mean return to annual
    if 'Mean Return' in annual_metrics.columns:
        annual_metrics['Mean Return (Annual)'] = (1 + annual_metrics['Mean Return']) ** 252 - 1
        annual_metrics = annual_metrics.drop('Mean Return', axis=1)
    
    # Convert standard deviation to annual
    if 'Standard Deviation' in annual_metrics.columns:
        annual_metrics['Standard Deviation (Annual)'] = annual_metrics['Standard Deviation'] * np.sqrt(252)
        annual_metrics = annual_metrics.drop('Standard Deviation', axis=1)
    
    return annual_metrics

def visualize_network(correlation_matrix: pd.DataFrame, threshold: float = 0.5):
    """
    Visualizes the network of stocks based on correlations.
    
    Args:
        correlation_matrix: DataFrame with correlation matrix
        threshold: Correlation threshold for drawing edges
    """
    try:
        import networkx as nx
        
        # Create a graph
        G = nx.Graph()
        
        # Add nodes
        for ticker in correlation_matrix.columns:
            G.add_node(ticker)
        
        # Add edges if correlation is above threshold
        for i, ticker1 in enumerate(correlation_matrix.columns):
            for j, ticker2 in enumerate(correlation_matrix.columns):
                if i < j:  # Only add each edge once
                    correlation = correlation_matrix.loc[ticker1, ticker2]
                    if abs(correlation) > threshold:
                        G.add_edge(ticker1, ticker2, weight=correlation)
        
        # Calculate node positions using a spring layout
        pos = nx.spring_layout(G, seed=42)
        
        # Draw the network
        plt.figure(figsize=(12, 10))
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=100, alpha=0.8)
        
        # Draw edges with colors based on correlation
        edges = list(G.edges(data=True))
        edge_colors = [e[2]['weight'] for e in edges]
        nx.draw_networkx_edges(
            G, pos, edgelist=[(e[0], e[1]) for e in edges],
            width=2, alpha=0.5, edge_color=edge_colors, edge_cmap=plt.cm.coolwarm
        )
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
        
        plt.title('Stock Correlation Network')
        plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.coolwarm), label='Correlation')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("NetworkX is required for network visualization. Install it with: pip install networkx")

if __name__ == "__main__":
    # Example usage
    from data_preprocessing import prepare_stock_data
    from correlation_matrix import compute_correlation_matrix
    from clustering import cluster_stocks
    
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
    
    # Plot correlation matrix with clusters
    plot_clusters(corr_matrix, labels, residual_returns.columns.tolist())
    
    # Calculate and print Sharpe and Sortino ratios for a sample series
    sample_returns = residual_returns['AAPL']
    sharpe = calculate_sharpe_ratio(sample_returns)
    sortino = calculate_sortino_ratio(sample_returns)
    
    print(f"AAPL Sharpe Ratio: {sharpe:.4f}")
    print(f"AAPL Sortino Ratio: {sortino:.4f}")
    
    # Get eigenvalue distribution
    eigenvalues, variance_ratios = get_eigenvalue_distribution(corr_matrix)
    
    # Visualize the network
    visualize_network(corr_matrix, threshold=0.5)
