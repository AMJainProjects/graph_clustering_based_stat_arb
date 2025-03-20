import numpy as np
import pandas as pd
from typing import Tuple

def compute_correlation_matrix(
    returns: pd.DataFrame,
    window: int = 60
) -> pd.DataFrame:
    """
    Computes the rolling correlation matrix of stock returns.
    
    Args:
        returns: DataFrame with stock returns (can be raw or residual)
        window: Rolling window size in days
        
    Returns:
        DataFrame with correlation matrix for the last date
    """
    correlation_matrix = returns.iloc[-window:].corr()
    return correlation_matrix

def compute_correlation_matrix_for_date(
    returns: pd.DataFrame,
    date: str,
    window: int = 60
) -> pd.DataFrame:
    """
    Computes the correlation matrix for a specific date using a lookback window.
    
    Args:
        returns: DataFrame with stock returns (can be raw or residual)
        date: Target date in the index of returns DataFrame
        window: Lookback window size in days
        
    Returns:
        DataFrame with correlation matrix
    """
    # Get the index of the target date
    try:
        date_idx = returns.index.get_loc(date)
    except KeyError:
        raise ValueError(f"Date {date} not found in returns index")
    
    # Ensure we have enough data for the lookback window
    if date_idx < window:
        raise ValueError(f"Not enough data for {window}-day lookback from {date}")
    
    # Get the window of returns leading up to the target date
    window_returns = returns.iloc[date_idx - window + 1:date_idx + 1]
    
    # Compute the correlation matrix
    correlation_matrix = window_returns.corr()
    
    return correlation_matrix

def decompose_correlation_matrix(
    correlation_matrix: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Decomposes the correlation matrix into positive and negative components.
    
    Args:
        correlation_matrix: DataFrame with correlation matrix
        
    Returns:
        Tuple of (positive_matrix, negative_matrix) DataFrames
    """
    # Create copies to avoid modifying the original
    positive_matrix = correlation_matrix.copy()
    negative_matrix = correlation_matrix.copy()
    
    # Zero out negative/positive values
    positive_matrix[positive_matrix < 0] = 0
    negative_matrix[negative_matrix > 0] = 0
    
    # Take absolute value of negative matrix for easier handling
    negative_matrix = negative_matrix.abs()
    
    return positive_matrix, negative_matrix

def compute_correlation_matrix_over_time(
    returns: pd.DataFrame,
    window: int = 60,
    step: int = 1
) -> dict:
    """
    Computes correlation matrices for a series of dates in the returns DataFrame.
    
    Args:
        returns: DataFrame with stock returns
        window: Lookback window size in days
        step: Number of days between each correlation matrix calculation
        
    Returns:
        Dictionary mapping dates to correlation matrices
    """
    correlation_matrices = {}
    
    # Start from the first date where we have enough lookback data
    start_idx = window - 1
    
    # Iterate through dates with the specified step
    for i in range(start_idx, len(returns.index), step):
        date = returns.index[i]
        try:
            # Compute correlation matrix for this date
            corr_matrix = compute_correlation_matrix_for_date(
                returns, date, window
            )
            correlation_matrices[date] = corr_matrix
        except ValueError as e:
            print(f"Warning: {e}")
            continue
    
    return correlation_matrices

def compute_degree_matrix(
    adjacency_matrix: pd.DataFrame,
    signed: bool = False
) -> pd.DataFrame:
    """
    Computes the degree matrix from an adjacency matrix.
    
    Args:
        adjacency_matrix: DataFrame with adjacency matrix (can be correlation matrix)
        signed: Whether to take absolute values for signed matrices
        
    Returns:
        DataFrame with degree matrix (diagonal matrix)
    """
    # Create a copy of the index and columns for the result
    index = adjacency_matrix.index
    columns = adjacency_matrix.columns
    
    # Initialize a diagonal matrix with zeros
    degree_matrix = pd.DataFrame(0, index=index, columns=columns)
    
    # Calculate degrees
    if signed:
        # For signed networks, sum absolute values
        degrees = adjacency_matrix.abs().sum(axis=1)
    else:
        # For unsigned networks, just sum
        degrees = adjacency_matrix.sum(axis=1)
    
    # Set diagonal elements to degrees
    for i, idx in enumerate(index):
        degree_matrix.loc[idx, idx] = degrees[idx]
    
    return degree_matrix

def normalize_by_degree(
    matrix: pd.DataFrame,
    degree_matrix: pd.DataFrame,
    method: str = 'symmetric'
) -> pd.DataFrame:
    """
    Normalizes a matrix using the degree matrix.
    
    Args:
        matrix: DataFrame to normalize
        degree_matrix: Diagonal degree matrix
        method: Normalization method ('symmetric' or 'random_walk')
        
    Returns:
        Normalized DataFrame
    """
    # Extract the degrees (diagonal of the degree matrix)
    degrees = np.diag(degree_matrix.values).copy()
    
    # Ensure no division by zero
    degrees[degrees == 0] = 1e-10
    
    if method == 'symmetric':
        # Symmetric normalization: D^(-1/2) * M * D^(-1/2)
        deg_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
        result = deg_inv_sqrt @ matrix.values @ deg_inv_sqrt
    elif method == 'random_walk':
        # Random walk normalization: D^(-1) * M
        deg_inv = np.diag(1.0 / degrees)
        result = deg_inv @ matrix.values
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    # Convert back to DataFrame with original indices
    normalized = pd.DataFrame(
        result,
        index=matrix.index,
        columns=matrix.columns
    )
    
    return normalized

if __name__ == "__main__":
    # Example usage
    from data_preprocessing import prepare_stock_data
    
    tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'PG']
    start_date = '2020-01-01'
    end_date = '2022-12-31'
    
    returns, betas, residual_returns = prepare_stock_data(tickers, start_date, end_date)
    
    # Compute correlation matrix for the last date
    corr_matrix = compute_correlation_matrix(residual_returns)
    print("Correlation Matrix:")
    print(corr_matrix)
    
    # Decompose into positive and negative components
    pos_matrix, neg_matrix = decompose_correlation_matrix(corr_matrix)
    print("\nPositive Matrix:")
    print(pos_matrix)
    print("\nNegative Matrix:")
    print(neg_matrix)
    
    # Compute degree matrix
    degree_matrix = compute_degree_matrix(corr_matrix, signed=True)
    print("\nDegree Matrix:")
    print(degree_matrix)
