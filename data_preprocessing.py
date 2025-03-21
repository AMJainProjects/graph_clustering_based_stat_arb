import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
from typing import Tuple, List, Optional, Dict
import pandas_datareader.data as web
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def fetch_stock_data(
        tickers: List[str],
        start_date: str,
        end_date: str,
        include_market: bool = True,
        batch_size: int = 10  # Added parameter for batch processing
) -> pd.DataFrame:
    """
    Fetches historical stock price data for the given tickers.
    Implements batch processing to handle large lists of tickers.

    Args:
        tickers: List of stock ticker symbols
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        include_market: Whether to include SPY as market proxy
        batch_size: Number of tickers to fetch in each batch

    Returns:
        DataFrame with daily adjusted closing prices for each ticker
    """
    # Add SPY as market proxy if not already included
    if include_market and 'SPY' not in tickers:
        all_tickers = tickers + ['SPY']
    else:
        all_tickers = tickers.copy()

    # Initialize an empty DataFrame for prices
    all_prices = pd.DataFrame()

    # Process tickers in batches
    for i in range(0, len(all_tickers), batch_size):
        batch_tickers = all_tickers[i:i + batch_size]
        print(f"Fetching batch {i // batch_size + 1} with {len(batch_tickers)} tickers")

        try:
            # Fetch data for this batch
            batch_data = yf.download(
                batch_tickers,
                start=start_date,
                end=end_date,
                progress=True,
                auto_adjust=False,
                threads=False
            )

            # Extract adjusted close prices
            if len(batch_tickers) == 1:
                # Handle the case of a single ticker (which returns a different format)
                ticker = batch_tickers[0]
                batch_prices = batch_data['Adj Close'].to_frame(ticker)
            else:
                batch_prices = batch_data['Adj Close']

            # Merge with the main DataFrame
            if all_prices.empty:
                all_prices = batch_prices
            else:
                all_prices = all_prices.join(batch_prices, how='outer')

        except Exception as e:
            print(f"Error fetching batch {i // batch_size + 1}: {e}")
            # Continue with the next batch instead of failing completely
            continue

    # Forward fill any missing values (e.g., non-trading days)
    all_prices = all_prices.ffill()

    if all_prices.empty:
        raise ValueError("Failed to fetch any stock data")

    return all_prices


def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates daily returns from price data.

    Args:
        prices: DataFrame with daily prices

    Returns:
        DataFrame with daily returns
    """
    returns = prices.pct_change().ffill().bfill()
    return returns


def get_market_cap_data(
        tickers: List[str],
        date: Optional[str] = None
) -> pd.DataFrame:
    """
    Gets the market capitalization data for the tickers.

    Args:
        tickers: List of stock ticker symbols
        date: Optional date to get market cap for (defaults to latest)

    Returns:
        DataFrame with market cap data
    """
    market_caps = {}

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            if date:
                # Get historical market cap if date provided
                hist_data = stock.history(period="1d", start=date, end=date)
                if not hist_data.empty:
                    price = hist_data['Close'].iloc[0]
                    shares = stock.info.get('sharesOutstanding', None)
                    if shares:
                        market_caps[ticker] = price * shares
            else:
                # Get current market cap
                market_caps[ticker] = stock.info.get('marketCap', None)
        except:
            market_caps[ticker] = None

    return pd.DataFrame({'marketCap': market_caps})


def filter_stocks_by_market_cap(
        tickers: List[str],
        percentile: float = 0.75,
        date: Optional[str] = None
) -> List[str]:
    """
    Filters stocks to keep only those in the top percentile of market cap.

    Args:
        tickers: List of stock ticker symbols
        percentile: Percentile threshold (e.g., 0.75 for top 25%)
        date: Optional date to calculate market cap for

    Returns:
        Filtered list of ticker symbols
    """
    market_caps = get_market_cap_data(tickers, date)
    market_caps = market_caps.ffill().bfill()

    threshold = market_caps['marketCap'].quantile(percentile)
    top_tickers = market_caps[market_caps['marketCap'] >= threshold].index.tolist()

    return top_tickers


def calculate_betas(
        returns: pd.DataFrame,
        market_ticker: str = 'SPY',
        window: int = 60,
        include_market: bool = True  # New parameter to include market ticker
) -> pd.DataFrame:
    """
    Calculates rolling betas for each stock against the market.

    Args:
        returns: DataFrame with daily returns
        market_ticker: Market proxy ticker symbol
        window: Rolling window size in days
        include_market: Whether to include the market ticker in the output

    Returns:
        DataFrame with beta values for each stock and date
    """
    # Determine which columns to include in the output
    if include_market:
        # Include all columns
        output_columns = returns.columns
    else:
        # Exclude the market ticker (traditional approach)
        output_columns = returns.columns.drop(market_ticker)

    # Initialize the betas DataFrame
    betas = pd.DataFrame(index=returns.index[window:], columns=output_columns)

    # Get market returns
    market_returns = returns[market_ticker]

    # Calculate rolling betas for each stock
    for ticker in output_columns:
        if ticker == market_ticker and include_market:
            # Set beta = 1 for the market itself
            betas[ticker] = 1.0
        else:
            stock_returns = returns[ticker]

            # Calculate rolling betas using covariance and variance
            rolling_cov = stock_returns.rolling(window=window).cov(market_returns)
            rolling_var = market_returns.rolling(window=window).var()

            rolling_beta = rolling_cov / rolling_var
            betas[ticker] = rolling_beta[window:]

    return betas


def calculate_residual_returns(
        returns: pd.DataFrame,
        betas: pd.DataFrame,
        market_ticker: str = 'SPY'
) -> pd.DataFrame:
    """
    Calculates residual returns by removing the market component.

    Args:
        returns: DataFrame with daily returns
        betas: DataFrame with beta values
        market_ticker: Market proxy ticker symbol

    Returns:
        DataFrame with residual returns
    """
    # Initialize residual returns with the same columns as betas
    residual_returns = pd.DataFrame(index=betas.index, columns=betas.columns)

    # Get market returns for the same dates as in betas
    market_returns = returns.loc[betas.index, market_ticker]

    # Calculate residual returns for each stock
    for ticker in betas.columns:
        if ticker == market_ticker:
            # For the market itself, residual returns should be 0
            # (perfect correlation with itself means no unexplained component)
            residual_returns[ticker] = 0.0
        else:
            stock_returns = returns.loc[betas.index, ticker]
            stock_betas = betas[ticker]

            # Calculate residual returns: R_res = R - beta * R_mkt
            residual_returns[ticker] = stock_returns - stock_betas * market_returns

    return residual_returns


def prepare_stock_data(
        tickers: List[str],
        start_date: str,
        end_date: str,
        beta_window: int = 60,
        market_cap_percentile: float = 0.75,
        apply_market_cap_filter: bool = False,  # Added parameter to make filtering optional
        include_market_in_results: bool = True  # New parameter to include SPY in results
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    End-to-end function to prepare stock data for analysis.

    Args:
        tickers: List of stock ticker symbols
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        beta_window: Window for beta calculation in days
        market_cap_percentile: Percentile for market cap filtering
        apply_market_cap_filter: Whether to apply market cap filtering (default: False)
        include_market_in_results: Whether to include market (SPY) in betas and residuals

    Returns:
        Tuple of (returns, betas, residual_returns) DataFrames
    """
    # Filter stocks by market cap only if requested
    if apply_market_cap_filter:
        filtered_tickers = filter_stocks_by_market_cap(tickers, market_cap_percentile)
        print(f"Market cap filter applied. Kept {len(filtered_tickers)} out of {len(tickers)} tickers.")
    else:
        filtered_tickers = tickers
        print(f"No market cap filter applied. Using all {len(tickers)} tickers.")

    # Fetch data with improved error handling
    try:
        prices = fetch_stock_data(filtered_tickers, start_date, end_date)
        print(f"Successfully fetched data for {len(prices.columns)} out of {len(filtered_tickers)} tickers.")
    except Exception as e:
        print(f"Error in fetch_stock_data: {e}")
        raise

    # Calculate returns
    returns = calculate_returns(prices)

    # Calculate betas (now with option to include market ticker)
    betas = calculate_betas(returns, window=beta_window, include_market=include_market_in_results)

    # Calculate residual returns
    residual_returns = calculate_residual_returns(returns, betas)

    return returns, betas, residual_returns

# ==================== NEW ADVANCED FACTOR MODELS ====================

def fetch_fama_french_factors(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches Fama-French 5-factor data from Kenneth French's data library.

    Args:
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format

    Returns:
        DataFrame with factor returns
    """
    try:
        # Try to fetch Fama-French 5-factor data
        ff_data = web.DataReader(
            'F-F_Research_Data_5_Factors_2x3_daily',
            'famafrench',
            start=start_date,
            end=end_date
        )[0]

        # Convert from percent to decimal
        ff_data = ff_data / 100.0

        # Rename columns
        ff_data.columns = ['MKT-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']

        # Calculate market return
        ff_data['MKT'] = ff_data['MKT-RF'] + ff_data['RF']

        return ff_data

    except Exception as e:
        print(f"Error fetching Fama-French data: {e}")
        # Create mock data as fallback
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')

        # Create mock factors with realistic variances
        mock_data = pd.DataFrame({
            'MKT-RF': np.random.normal(0.0005, 0.01, len(date_range)),
            'SMB': np.random.normal(0.0001, 0.005, len(date_range)),
            'HML': np.random.normal(0.0001, 0.005, len(date_range)),
            'RMW': np.random.normal(0.0001, 0.003, len(date_range)),
            'CMA': np.random.normal(0.0001, 0.003, len(date_range)),
            'RF': np.ones(len(date_range)) * 0.0001  # ~2.5% annual risk-free rate
        }, index=date_range)

        mock_data['MKT'] = mock_data['MKT-RF'] + mock_data['RF']

        return mock_data


def calculate_multifactor_residuals(
        returns: pd.DataFrame,
        factor_returns: pd.DataFrame,
        rolling_window: Optional[int] = None,
        market_ticker: str = 'SPY',
        include_market: bool = True
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Calculates residual returns using a multi-factor model.

    Args:
        returns: DataFrame with stock returns
        factor_returns: DataFrame with factor returns
        rolling_window: Optional window for rolling regression
        market_ticker: Market proxy ticker symbol
        include_market: Whether to include the market ticker in output

    Returns:
        Tuple of (residual_returns, factor_betas)
    """
    # Align data
    common_dates = returns.index.intersection(factor_returns.index)
    aligned_returns = returns.loc[common_dates]
    aligned_factors = factor_returns.loc[common_dates]

    # Determine which columns to include in the output
    if include_market and market_ticker in returns.columns:
        output_columns = returns.columns
    else:
        output_columns = returns.columns.drop(market_ticker) if market_ticker in returns.columns else returns.columns

    # Initialize output DataFrames
    residual_returns = pd.DataFrame(index=common_dates, columns=output_columns)
    factor_betas = {ticker: pd.DataFrame(index=common_dates, columns=factor_returns.columns)
                    for ticker in output_columns}

    # Prepare factors (add constant)
    X = sm.add_constant(aligned_factors)

    # Special handling for market ticker if it's included
    if include_market and market_ticker in returns.columns:
        # If 'MKT' is one of the factors, the market ticker should be highly correlated with it
        if 'MKT' in factor_returns.columns:
            # For the market itself, set the MKT beta to 1.0 and others to 0.0
            for factor in factor_returns.columns:
                factor_betas[market_ticker][factor] = 1.0 if factor == 'MKT' else 0.0

            # Set residual returns for market to 0 (perfectly explained by the MKT factor)
            residual_returns[market_ticker] = 0.0

    if rolling_window is None:
        # Static regression for the entire period
        for ticker in output_columns:
            # Skip market ticker if it's been handled above
            if ticker == market_ticker and include_market and 'MKT' in factor_returns.columns:
                continue

            y = aligned_returns[ticker]

            # Fit linear model
            try:
                model = sm.OLS(y, X).fit()

                # Store betas
                for i, factor in enumerate(aligned_factors.columns):
                    factor_betas[ticker][factor] = model.params[i + 1]  # Skip constant

                # Calculate residual returns
                predicted = model.predict(X)
                residual_returns[ticker] = y - predicted
            except Exception as e:
                print(f"Error in regression for {ticker}: {e}")
                residual_returns[ticker] = aligned_returns[ticker]

    else:
        # Rolling regression
        for ticker in output_columns:
            # Skip market ticker if it's been handled above
            if ticker == market_ticker and include_market and 'MKT' in factor_returns.columns:
                continue

            y = aligned_returns[ticker]

            # Initialize with NaN
            for factor in aligned_factors.columns:
                factor_betas[ticker][factor] = np.nan

            # Calculate rolling regression
            for i in range(rolling_window, len(common_dates) + 1):
                window_start = i - rolling_window
                window_end = i

                window_X = X.iloc[window_start:window_end]
                window_y = y.iloc[window_start:window_end]

                try:
                    # Fit linear model for this window
                    model = sm.OLS(window_y, window_X).fit()

                    # Store betas for the last date in the window
                    current_date = common_dates[i - 1]
                    for j, factor in enumerate(aligned_factors.columns):
                        factor_betas[ticker].loc[current_date, factor] = model.params[j + 1]  # Skip constant

                    # Calculate residual return for the last date
                    predicted = model.predict(X.iloc[i - 1:i])
                    residual_returns.loc[current_date, ticker] = y.iloc[i - 1] - predicted.iloc[0]

                except Exception as e:
                    if i == len(common_dates):
                        print(f"Error in rolling regression for {ticker} at date {common_dates[i - 1]}: {e}")
                    # Keep raw return if regression fails
                    current_date = common_dates[i - 1]
                    residual_returns.loc[current_date, ticker] = y.iloc[i - 1]

        # Forward fill betas
        for ticker in output_columns:
            factor_betas[ticker] = factor_betas[ticker].ffill().bfill()

    return residual_returns, factor_betas


def extract_pca_factors(
        returns: pd.DataFrame,
        n_components: Optional[int] = None,
        explained_variance_threshold: float = 0.9,
        center: bool = True,
        scale: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, PCA]:
    """
    Extracts latent factors using PCA.

    Args:
        returns: DataFrame with stock returns
        n_components: Number of components to extract
        explained_variance_threshold: Minimum explained variance threshold
        center: Whether to center the data
        scale: Whether to scale the data

    Returns:
        Tuple of (factor_returns, loadings, explained_variance, pca_model)
    """
    # Handle missing values
    returns_filled = returns.fillna(0)

    # Standardize returns if requested
    if center or scale:
        scaler = StandardScaler(with_mean=center, with_std=scale)
        scaled_returns = scaler.fit_transform(returns_filled)
    else:
        scaled_returns = returns_filled.values

    # Determine number of components based on explained variance if not provided
    if n_components is None and explained_variance_threshold:
        # First run PCA to determine number of components
        pca_full = PCA()
        pca_full.fit(scaled_returns)

        # Find number of components needed
        explained_var_ratio = np.cumsum(pca_full.explained_variance_ratio_)
        n_components = np.argmax(explained_var_ratio >= explained_variance_threshold) + 1

        # Ensure at least 1 component
        n_components = max(1, n_components)

    # Extract components
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(scaled_returns)

    # Create DataFrame of factor returns
    factor_returns = pd.DataFrame(
        principal_components,
        index=returns.index,
        columns=[f'PC{i + 1}' for i in range(n_components)]
    )

    # Get factor loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        index=returns.columns,
        columns=[f'PC{i + 1}' for i in range(n_components)]
    )

    # Calculate variance explained
    explained_variance = pd.Series(
        pca.explained_variance_ratio_,
        index=[f'PC{i + 1}' for i in range(n_components)]
    )

    return factor_returns, loadings, explained_variance, pca


def calculate_pca_residuals(
        returns: pd.DataFrame,
        n_components: Optional[int] = None,
        explained_variance_threshold: float = 0.9
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Calculates residual returns by removing principal components.

    Args:
        returns: DataFrame with stock returns
        n_components: Number of components to extract
        explained_variance_threshold: Minimum explained variance threshold

    Returns:
        Tuple of (residual_returns, factor_returns, loadings, explained_variance)
    """
    # Extract PCA factors
    factor_returns, loadings, explained_variance, pca = extract_pca_factors(
        returns, n_components, explained_variance_threshold
    )

    # Calculate residual returns
    X = sm.add_constant(factor_returns)
    residual_returns = pd.DataFrame(index=returns.index, columns=returns.columns)
    factor_betas = pd.DataFrame(index=returns.columns, columns=factor_returns.columns)

    for ticker in returns.columns:
        y = returns[ticker].fillna(0)  # Replace NaN with zeros for regression

        try:
            # Fit linear model
            model = sm.OLS(y, X).fit()

            # Store betas
            for factor in factor_returns.columns:
                factor_betas.loc[ticker, factor] = model.params[factor]

            # Calculate residual returns: actual - predicted
            predicted = model.predict(X)
            residual_returns[ticker] = y - predicted
        except Exception as e:
            print(f"Error in PCA regression for {ticker}: {e}")
            residual_returns[ticker] = returns[ticker]

    return residual_returns, factor_returns, factor_betas, explained_variance


def prepare_multifactor_data(
        tickers: List[str],
        start_date: str,
        end_date: str,
        factor_model: str = 'fama_french',
        use_pca: bool = False,
        n_components: Optional[int] = None,
        rolling_window: Optional[int] = None,
        market_cap_percentile: float = 0.75,
        apply_market_cap_filter: bool = False,  # Added parameter
        include_market_in_results: bool = True  # Added parameter
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """
    End-to-end function to prepare stock data using a multi-factor model.

    Args:
        tickers: List of stock ticker symbols
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        factor_model: Type of factor model ('fama_french', 'pca', 'capm')
        use_pca: Whether to use PCA in addition to specified factor model
        n_components: Number of PCA components (if use_pca is True)
        rolling_window: Window for rolling regression
        market_cap_percentile: Percentile for market cap filtering
        apply_market_cap_filter: Whether to apply market cap filtering
        include_market_in_results: Whether to include market (SPY) in results

    Returns:
        Tuple of (returns, factor_returns, residual_returns, factor_data)
    """
    # Filter stocks by market cap if requested
    if apply_market_cap_filter:
        filtered_tickers = filter_stocks_by_market_cap(tickers, market_cap_percentile)
        print(f"Market cap filter applied. Kept {len(filtered_tickers)} out of {len(tickers)} tickers.")
    else:
        filtered_tickers = tickers
        print(f"No market cap filter applied. Using all {len(tickers)} tickers.")

    # Fetch price data
    prices = fetch_stock_data(filtered_tickers, start_date, end_date)

    # Calculate returns
    returns = calculate_returns(prices)

    # Prepare additional data based on factor model
    factor_data = {}

    if factor_model == 'capm':
        # CAPM (single-factor market model)
        market_ticker = 'SPY'
        betas = calculate_betas(returns, market_ticker=market_ticker,
                                window=rolling_window or 60,
                                include_market=include_market_in_results)
        residual_returns = calculate_residual_returns(returns, betas, market_ticker=market_ticker)

        # Create factor returns DataFrame with just market
        factor_returns = pd.DataFrame({'MKT': returns[market_ticker]})
        factor_data['betas'] = betas

    elif factor_model == 'fama_french':
        # Fama-French multi-factor model
        # Need to fetch data a bit earlier for rolling window
        adjusted_start = pd.Timestamp(start_date) - pd.Timedelta(days=rolling_window * 2 if rolling_window else 0)
        ff_factors = fetch_fama_french_factors(adjusted_start.strftime('%Y-%m-%d'), end_date)

        # Calculate residuals using Fama-French factors
        if include_market_in_results:
            # Keep SPY in the data for calculation
            ticker_returns = returns
        else:
            # Drop SPY for traditional calculation
            ticker_returns = returns.drop('SPY', axis=1) if 'SPY' in returns.columns else returns

        residual_returns, factor_betas = calculate_multifactor_residuals(
            ticker_returns,
            ff_factors,
            rolling_window
        )

        factor_returns = ff_factors
        factor_data['betas'] = factor_betas

    elif factor_model == 'pca':
        # Pure PCA model
        if include_market_in_results:
            # Keep SPY in the data for PCA
            ticker_returns = returns
        else:
            # Traditional approach: exclude market from PCA
            ticker_returns = returns.drop('SPY', axis=1) if 'SPY' in returns.columns else returns

        residual_returns, factor_returns, factor_betas, explained_variance = calculate_pca_residuals(
            ticker_returns, n_components
        )

        factor_data['betas'] = factor_betas
        factor_data['explained_variance'] = explained_variance

    else:
        raise ValueError(f"Unknown factor model: {factor_model}")

    # Add PCA on top of existing factor model if requested
    if use_pca and factor_model != 'pca':
        # Apply PCA to the residuals from the first-stage factor model
        second_stage_residuals, pca_factors, pca_betas, pca_variance = calculate_pca_residuals(
            residual_returns, n_components
        )

        # Update residuals
        residual_returns = second_stage_residuals

        # Store PCA information
        factor_data['pca_factors'] = pca_factors
        factor_data['pca_betas'] = pca_betas
        factor_data['pca_variance'] = pca_variance

    return returns, factor_returns, residual_returns, factor_data


if __name__ == "__main__":
    # Example usage
    tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'PG']
    start_date = '2020-01-01'
    end_date = '2022-12-31'

    # Traditional CAPM approach
    print("Testing CAPM approach:")
    returns, betas, residual_returns = prepare_stock_data(tickers, start_date, end_date)

    print("Returns shape:", returns.shape)
    print("Betas shape:", betas.shape)
    print("Residual returns shape:", residual_returns.shape)

    print("\nSample of residual returns:")
    print(residual_returns.head())

    # Multi-factor approach
    print("\nTesting multi-factor approach:")
    ff_returns, ff_factors, ff_residuals, ff_data = prepare_multifactor_data(
        tickers, start_date, end_date, factor_model='fama_french'
    )

    print("Returns shape:", ff_returns.shape)
    print("Factor returns shape:", ff_factors.shape)
    print("Residual returns shape:", ff_residuals.shape)

    # PCA approach
    print("\nTesting PCA approach:")
    pca_returns, pca_factors, pca_residuals, pca_data = prepare_multifactor_data(
        tickers, start_date, end_date, factor_model='pca', n_components=5
    )

    print("Returns shape:", pca_returns.shape)
    print("PCA factor returns shape:", pca_factors.shape)
    print("PCA residual returns shape:", pca_residuals.shape)
    print("Explained variance:", pca_data['explained_variance'])