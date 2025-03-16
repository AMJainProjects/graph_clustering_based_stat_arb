"""
Configuration settings for the statistical arbitrage strategy.
"""

# Data settings
DATA_SETTINGS = {
    'start_date': '2000-01-01',  # Start date for backtest
    'end_date': '2022-12-31',    # End date for backtest
    'market_ticker': 'SPY',      # Market proxy ticker
    'market_cap_percentile': 0.75,  # Filter for top 25% market cap stocks
    'min_stocks_per_cluster': 2  # Minimum number of stocks required in a cluster
}

# Beta calculation settings
BETA_SETTINGS = {
    'window': 60  # Rolling window size in days for beta calculation
}

# Correlation matrix settings
CORRELATION_SETTINGS = {
    'window': 20  # Rolling window size in days for correlation calculation
}

# Clustering settings
CLUSTERING_SETTINGS = {
    'methods': [
        'spectral',           # Standard spectral clustering
        'signed_laplacian_sym',  # Signed Laplacian with symmetric normalization
        'signed_laplacian_rw',   # Signed Laplacian with random walk normalization
        'sponge',                # SPONGE algorithm
        'sponge_sym'             # SPONGEsym algorithm
    ],
    'default_method': 'sponge_sym',  # Default method to use
    'n_clusters_methods': [
        'mp',   # Marchenko-Pastur distribution
        'var',  # Variance explained (90%)
        'fixed'  # Fixed number of clusters
    ],
    'default_n_clusters_method': 'mp',  # Default method to determine number of clusters
    'fixed_n_clusters': 30,  # Fixed number of clusters (if using 'fixed' method)
    'variance_threshold': 0.9,  # Variance threshold for 'var' method (90%)
    'sponge_tau_p': 0.5,  # Regularization parameter for positive matrix in SPONGE
    'sponge_tau_n': 0.5   # Regularization parameter for negative matrix in SPONGE
}

# Portfolio construction settings
PORTFOLIO_SETTINGS = {
    'lookback_window': 5,  # Days to look back for winner/loser identification
    'rebalance_period': 3,  # Days between portfolio rebalancing
    'threshold': 0.0,  # Threshold for classifying as winner/loser
    'stop_win_threshold': 0.05  # Stop-win threshold (5%)
}

# Performance evaluation settings
PERFORMANCE_SETTINGS = {
    'risk_free_rate': 0.0,  # Risk-free rate (annual)
    'benchmark': 'SPY'      # Benchmark ticker
}

# Universe settings - different stock universes to test
UNIVERSES = {
    'sp500': {
        'name': 'S&P 500 Components',
        'filepath': 'data/sp500_components.csv',  # CSV file with tickers
        'source': 'yfinance'  # Data source
    },
    'tech': {
        'name': 'Technology Stocks',
        'tickers': [
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'META', 'AMZN', 'TSLA', 'NVDA', 
            'INTC', 'AMD', 'ADBE', 'CRM', 'CSCO', 'ORCL', 'IBM', 'QCOM', 
            'TXN', 'AVGO', 'PYPL', 'NFLX'
        ],
        'source': 'yfinance'  # Data source
    },
    'finance': {
        'name': 'Financial Stocks',
        'tickers': [
            'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'V', 'MA', 'AXP', 'DFS',
            'COF', 'BLK', 'SCHW', 'BK', 'PNC', 'TFC', 'USB', 'AIG', 'MET', 'PRU'
        ],
        'source': 'yfinance'  # Data source
    },
    'healthcare': {
        'name': 'Healthcare Stocks',
        'tickers': [
            'JNJ', 'PFE', 'MRK', 'ABT', 'ABBV', 'LLY', 'BMY', 'TMO', 'DHR', 'UNH',
            'CVS', 'AMGN', 'GILD', 'ISRG', 'BIIB', 'VRTX', 'REGN', 'ILMN', 'ZTS', 'BSX'
        ],
        'source': 'yfinance'  # Data source
    },
    'consumer': {
        'name': 'Consumer Stocks',
        'tickers': [
            'WMT', 'PG', 'KO', 'PEP', 'HD', 'MCD', 'SBUX', 'NKE', 'COST', 'TGT',
            'LOW', 'MDLZ', 'CL', 'EL', 'KHC', 'KR', 'SYY', 'GIS', 'HSY', 'K'
        ],
        'source': 'yfinance'  # Data source
    },
    'mixed': {
        'name': 'Mixed Sectors',
        'tickers': [
            'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'JPM', 'V', 'PG', 'JNJ',
            'WMT', 'DIS', 'HD', 'MCD', 'VZ', 'T', 'XOM', 'CVX', 'NEE', 'LIN',
            'UNH', 'PFE', 'KO', 'INTC', 'CSCO', 'NVDA', 'AMD', 'CRM', 'BA', 'CAT',
            'GE', 'MMM', 'HON', 'UNP', 'FDX', 'LMT', 'RTX', 'GS', 'MS', 'BAC'
        ],
        'source': 'yfinance'  # Data source
    }
}

# Default universe to use
DEFAULT_UNIVERSE = 'mixed'

# Visualization settings
VISUALIZATION_SETTINGS = {
    'figsize': (12, 8),
    'dpi': 100,
    'palette': 'tab10',  # Color palette for plots
    'correlation_cmap': 'coolwarm',  # Colormap for correlation matrices
    'save_plots': True,
    'plots_directory': 'plots/'
}

# Output settings
OUTPUT_SETTINGS = {
    'results_directory': 'results/',
    'save_results': True,
    'plot_format': 'png'
}
