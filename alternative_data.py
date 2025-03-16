import pandas as pd
import numpy as np
import requests
import json
import time
import yfinance as yf
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler
import datetime

class SentimentDataProcessor:
    """
    Processes and integrates sentiment data from various sources.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initializes the sentiment data processor.
        
        Args:
            api_key: Optional API key for sentiment data providers
        """
        self.api_key = api_key
    
    def fetch_news_sentiment(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        provider: str = 'mock'
    ) -> pd.DataFrame:
        """
        Fetches news sentiment data for the given tickers.
        
        Args:
            tickers: List of stock ticker symbols
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            provider: Sentiment data provider ('alphavantage', 'bloomberg', 'mock')
            
        Returns:
            DataFrame with daily sentiment scores
        """
        if provider == 'alphavantage':
            return self._fetch_alphavantage_sentiment(tickers, start_date, end_date)
        elif provider == 'bloomberg':
            return self._fetch_bloomberg_sentiment(tickers, start_date, end_date)
        elif provider == 'mock':
            return self._generate_mock_sentiment(tickers, start_date, end_date)
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def _fetch_alphavantage_sentiment(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Fetches sentiment data from Alpha Vantage API.
        
        Args:
            tickers: List of stock ticker symbols
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame with daily sentiment scores
        """
        if not self.api_key:
            raise ValueError("API key required for Alpha Vantage")
        
        sentiment_data = {}
        
        for ticker in tickers:
            try:
                endpoint = f"https://www.alphavantage.co/query"
                params = {
                    'function': 'NEWS_SENTIMENT',
                    'tickers': ticker,
                    'time_from': start_date,
                    'time_to': end_date,
                    'limit': 1000,
                    'apikey': self.api_key
                }
                
                response = requests.get(endpoint, params=params)
                data = response.json()
                
                # Process Alpha Vantage response
                if 'feed' not in data:
                    print(f"No sentiment data found for {ticker}")
                    continue
                
                # Extract sentiment scores from feed
                ticker_sentiment = {}
                
                for item in data['feed']:
                    time_published = item.get('time_published', '')[:10]  # Get YYYY-MM-DD
                    
                    if not time_published:
                        continue
                    
                    # Find ticker-specific sentiment in this news item
                    for ticker_sentiment_entry in item.get('ticker_sentiment', []):
                        if ticker_sentiment_entry.get('ticker') == ticker:
                            sentiment_score = float(ticker_sentiment_entry.get('ticker_sentiment_score', 0))
                            
                            # Aggregate by date (simple average)
                            if time_published in ticker_sentiment:
                                ticker_sentiment[time_published].append(sentiment_score)
                            else:
                                ticker_sentiment[time_published] = [sentiment_score]
                
                # Average sentiment scores by date
                sentiment_data[ticker] = {
                    date: np.mean(scores) for date, scores in ticker_sentiment.items()
                }
                
                # Rate limit compliance
                time.sleep(0.2)  # Alpha Vantage rate limits
            
            except Exception as e:
                print(f"Error fetching sentiment for {ticker}: {e}")
        
        # Convert to DataFrame
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        sentiment_df = pd.DataFrame(index=dates, columns=tickers)
        
        for ticker, ticker_data in sentiment_data.items():
            for date, score in ticker_data.items():
                try:
                    date_obj = pd.Timestamp(date)
                    if date_obj in sentiment_df.index:
                        sentiment_df.loc[date_obj, ticker] = score
                except:
                    continue
        
        # Forward fill and then backward fill missing values
        sentiment_df = sentiment_df.fillna(method='ffill').fillna(method='bfill')
        
        return sentiment_df
    
    def _fetch_bloomberg_sentiment(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Placeholder for Bloomberg sentiment data (requires subscription).
        
        Args:
            tickers: List of stock ticker symbols
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame with daily sentiment scores
        """
        # This would require Bloomberg API access
        # Fallback to mock data
        print("Bloomberg API not implemented. Using mock data instead.")
        return self._generate_mock_sentiment(tickers, start_date, end_date)
    
    def _generate_mock_sentiment(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Generates mock sentiment data for demonstration.
        
        Args:
            tickers: List of stock ticker symbols
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame with daily sentiment scores
        """
        # Create date range
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        
        # Initialize sentiment DataFrame
        sentiment_df = pd.DataFrame(index=dates, columns=tickers)
        
        # Generate mock sentiment scores
        for ticker in tickers:
            # Base sentiment random walk
            base_sentiment = np.cumsum(np.random.normal(0, 0.05, len(dates)))
            
            # Normalize to -1 to 1 range
            base_sentiment = (base_sentiment - base_sentiment.min()) / (base_sentiment.max() - base_sentiment.min()) * 2 - 1
            
            # Add some noise
            sentiment_df[ticker] = base_sentiment + np.random.normal(0, 0.1, len(dates))
            
            # Clip to [-1, 1] range
            sentiment_df[ticker] = sentiment_df[ticker].clip(-1, 1)
        
        return sentiment_df
    
    def fetch_social_sentiment(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        provider: str = 'mock',
        sources: List[str] = ['twitter', 'reddit', 'stocktwits']
    ) -> pd.DataFrame:
        """
        Fetches social media sentiment data.
        
        Args:
            tickers: List of stock ticker symbols
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            provider: Sentiment data provider ('stocktwits', 'twitter', 'mock')
            sources: List of social media sources to include
            
        Returns:
            DataFrame with daily sentiment scores
        """
        if provider == 'mock':
            return self._generate_mock_social_sentiment(tickers, start_date, end_date, sources)
        else:
            raise ValueError(f"Provider {provider} not implemented")
    
    def _generate_mock_social_sentiment(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        sources: List[str]
    ) -> Dict[str, pd.DataFrame]:
        """
        Generates mock social sentiment data.
        
        Args:
            tickers: List of stock ticker symbols
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            sources: List of social media sources to include
            
        Returns:
            Dictionary mapping sources to DataFrames with sentiment scores
        """
        # Create date range
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        
        # Initialize result dictionary
        result = {}
        
        for source in sources:
            # Initialize sentiment DataFrame for this source
            sentiment_df = pd.DataFrame(index=dates, columns=tickers)
            
            # Generate mock sentiment scores with source-specific characteristics
            for ticker in tickers:
                # Base sentiment - slowly varying component
                base_sentiment = np.cumsum(np.random.normal(0, 0.03, len(dates)))
                
                # Add source-specific patterns
                if source == 'twitter':
                    # Twitter - more volatile, weekly pattern
                    noise_scale = 0.2
                    # Add day-of-week effect (more activity on weekdays)
                    day_effect = np.array([0.1 if d.weekday() < 5 else -0.1 for d in dates])
                elif source == 'reddit':
                    # Reddit - more persistent trends
                    noise_scale = 0.15
                    # Add trend persistence
                    day_effect = np.repeat(np.random.normal(0, 0.3, len(dates) // 7 + 1), 7)[:len(dates)]
                elif source == 'stocktwits':
                    # StockTwits - very reactive to price changes
                    noise_scale = 0.25
                    # Simulate reaction to price changes
                    day_effect = np.random.normal(0, 0.2, len(dates))
                    # More outliers
                    outliers = np.random.choice(len(dates), size=int(len(dates) * 0.05), replace=False)
                    day_effect[outliers] *= 3
                else:
                    noise_scale = 0.1
                    day_effect = np.zeros(len(dates))
                
                # Combine components
                raw_sentiment = (base_sentiment + day_effect + np.random.normal(0, noise_scale, len(dates)))
                
                # Normalize to -1 to 1 range
                normalized_sentiment = (raw_sentiment - raw_sentiment.min()) / (raw_sentiment.max() - raw_sentiment.min()) * 2 - 1
                
                sentiment_df[ticker] = normalized_sentiment
            
            result[source] = sentiment_df
        
        return result
    
    def aggregate_sentiment(
        self,
        sentiment_data: Dict[str, pd.DataFrame],
        method: str = 'exponential_decay',
        decay_factor: float = 0.9,
        source_weights: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Aggregates sentiment scores from multiple sources.
        
        Args:
            sentiment_data: Dictionary mapping sources to sentiment DataFrames
            method: Aggregation method ('simple_average', 'exponential_decay')
            decay_factor: Weight decay factor for exponential weighting
            source_weights: Optional weights for different sources
            
        Returns:
            DataFrame with aggregated sentiment scores
        """
        if not sentiment_data:
            return pd.DataFrame()
        
        # Get a common set of dates and tickers
        sample_df = next(iter(sentiment_data.values()))
        dates = sample_df.index
        tickers = sample_df.columns
        
        # Initialize result DataFrame
        aggregated = pd.DataFrame(index=dates, columns=tickers)
        
        # Set default source weights if not provided
        if source_weights is None:
            source_weights = {source: 1.0 for source in sentiment_data.keys()}
        
        # Normalize source weights
        total_weight = sum(source_weights.values())
        normalized_weights = {s: w / total_weight for s, w in source_weights.items()}
        
        if method == 'simple_average':
            # Simple weighted average across sources
            for ticker in tickers:
                weighted_sentiment = pd.Series(0, index=dates)
                
                for source, sentiment_df in sentiment_data.items():
                    if ticker in sentiment_df.columns:
                        weight = normalized_weights.get(source, 0)
                        weighted_sentiment += sentiment_df[ticker] * weight
                
                aggregated[ticker] = weighted_sentiment
        
        elif method == 'exponential_decay':
            # Exponential decay weighting (more recent sentiment has higher weight)
            lookback_days = 10  # Number of days to look back
            
            for ticker in tickers:
                for i, date in enumerate(dates):
                    if i < lookback_days:
                        # Not enough history, use simple average
                        values = []
                        weights = []
                        
                        for source, sentiment_df in sentiment_data.items():
                            if ticker in sentiment_df.columns:
                                source_weight = normalized_weights.get(source, 0)
                                recent_sentiment = sentiment_df[ticker].iloc[:i+1]
                                values.extend(recent_sentiment.values)
                                weights.extend([source_weight] * len(recent_sentiment))
                        
                        if values:
                            aggregated.loc[date, ticker] = np.average(values, weights=weights)
                    else:
                        # Use exponential decay
                        values = []
                        weights = []
                        
                        for source, sentiment_df in sentiment_data.items():
                            if ticker in sentiment_df.columns:
                                source_weight = normalized_weights.get(source, 0)
                                recent_sentiment = sentiment_df[ticker].iloc[i-lookback_days:i+1]
                                
                                # Calculate time decay weights (newer = higher weight)
                                time_weights = np.array([decay_factor ** (lookback_days - d) 
                                                       for d in range(len(recent_sentiment))])
                                
                                # Combine source weight with time weights
                                combined_weights = time_weights * source_weight
                                
                                values.extend(recent_sentiment.values)
                                weights.extend(combined_weights)
                        
                        if values:
                            aggregated.loc[date, ticker] = np.average(values, weights=weights)
        
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
        
        return aggregated
    
    def calculate_sentiment_similarity(
        self,
        sentiment_scores: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculates similarity matrix based on sentiment patterns.
        
        Args:
            sentiment_scores: DataFrame with sentiment scores (tickers as columns)
            
        Returns:
            DataFrame with sentiment-based similarity matrix
        """
        # Calculate correlation matrix of sentiment scores
        sentiment_corr = sentiment_scores.corr()
        
        # Convert to similarity matrix (0-1 range, higher = more similar)
        similarity = (sentiment_corr + 1) / 2
        
        return similarity
    
    def integrate_with_clustering(
        self,
        sentiment_scores: pd.DataFrame,
        correlation_matrix: pd.DataFrame,
        alpha: float = 0.7
    ) -> pd.DataFrame:
        """
        Creates a hybrid distance matrix combining correlations and sentiment.
        
        Args:
            sentiment_scores: DataFrame with sentiment scores
            correlation_matrix: Correlation matrix from returns
            alpha: Weight for correlation (1-alpha for sentiment)
            
        Returns:
            Hybrid correlation/sentiment matrix
        """
        # Ensure we have sentiment data for all tickers in correlation matrix
        common_tickers = list(set(correlation_matrix.columns) & set(sentiment_scores.columns))
        
        if not common_tickers:
            print("No common tickers between correlation matrix and sentiment data")
            return correlation_matrix
        
        # Filter matrices to common tickers
        corr_filtered = correlation_matrix.loc[common_tickers, common_tickers]
        
        # Calculate sentiment similarity
        sentiment_similarity = self.calculate_sentiment_similarity(
            sentiment_scores[common_tickers]
        )
        
        # Combine matrices with weighting
        hybrid_matrix = alpha * corr_filtered + (1 - alpha) * sentiment_similarity
        
        return hybrid_matrix

class OptionsDataProcessor:
    """
    Processes and integrates options market data.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initializes the options data processor.
        
        Args:
            api_key: Optional API key for data providers
        """
        self.api_key = api_key
    
    def fetch_options_data(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        provider: str = 'yfinance'
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetches options chain data for the given tickers.
        
        Args:
            tickers: List of stock ticker symbols
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            provider: Data provider ('yfinance', 'mock')
            
        Returns:
            Dictionary mapping tickers to DataFrames with options data
        """
        if provider == 'yfinance':
            return self._fetch_yfinance_options(tickers)
        elif provider == 'mock':
            return self._generate_mock_options_data(tickers, start_date, end_date)
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def _fetch_yfinance_options(self, tickers: List[str]) -> Dict[str, Dict]:
        """
        Fetches current options data from Yahoo Finance.
        
        Args:
            tickers: List of stock ticker symbols
            
        Returns:
            Dictionary with options data
        """
        options_data = {}
        
        for ticker in tickers:
            try:
                # Get stock object
                stock = yf.Ticker(ticker)
                
                # Get all available expiration dates
                expirations = stock.options
                
                if not expirations:
                    print(f"No options data available for {ticker}")
                    continue
                
                # Dictionary to store calls and puts for each expiration
                ticker_options = {}
                
                for exp_date in expirations:
                    # Get option chain for this expiration
                    try:
                        chain = stock.option_chain(exp_date)
                        ticker_options[exp_date] = {
                            'calls': chain.calls,
                            'puts': chain.puts
                        }
                    except Exception as e:
                        print(f"Error fetching options for {ticker} expiring {exp_date}: {e}")
                
                options_data[ticker] = ticker_options
                
                # Rate limit compliance
                time.sleep(0.2)
            
            except Exception as e:
                print(f"Error fetching options for {ticker}: {e}")
        
        return options_data
    
    def _generate_mock_options_data(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str
    ) -> Dict[str, Dict]:
        """
        Generates mock options data for demonstration.
        
        Args:
            tickers: List of stock ticker symbols
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            Dictionary with mock options data
        """
        # Create random stock prices for the tickers
        stock_prices = {ticker: np.random.uniform(50, 500) for ticker in tickers}
        
        # Generate a set of expiration dates
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)
        
        # Generate monthly expirations
        expirations = []
        current_dt = start_dt
        while current_dt <= end_dt:
            # Third Friday of the month
            day = current_dt.replace(day=1)
            while day.weekday() != 4:  # Friday
                day = day + pd.Timedelta(days=1)
            friday = day + pd.Timedelta(weeks=2)
            expirations.append(friday.strftime('%Y-%m-%d'))
            current_dt = current_dt + pd.Timedelta(days=32)
            current_dt = current_dt.replace(day=1)
        
        # Generate weekly expirations
        weekly_expirations = []
        current_dt = start_dt
        while current_dt <= end_dt:
            # Find next Friday
            days_to_friday = (4 - current_dt.weekday()) % 7
            friday = current_dt + pd.Timedelta(days=days_to_friday)
            weekly_expirations.append(friday.strftime('%Y-%m-%d'))
            current_dt = friday + pd.Timedelta(days=7)
        
        # Combine expirations
        all_expirations = sorted(list(set(expirations + weekly_expirations)))
        
        # Generate options data for each ticker
        options_data = {}
        
        for ticker in tickers:
            ticker_price = stock_prices[ticker]
            ticker_options = {}
            
            for exp_date in all_expirations:
                exp_dt = pd.Timestamp(exp_date)
                days_to_expiry = (exp_dt - pd.Timestamp.now()).days
                if days_to_expiry < 0:
                    continue
                
                # Generate strikes (around current price)
                strike_step = ticker_price * 0.025  # 2.5% steps
                strikes = np.arange(
                    ticker_price * 0.7, 
                    ticker_price * 1.3 + strike_step, 
                    strike_step
                )
                
                # Create calls data
                calls_data = []
                for strike in strikes:
                    # Simple Black-Scholes approximation for IV
                    moneyness = ticker_price / strike
                    if moneyness > 1:  # ITM
                        iv = np.random.uniform(0.2, 0.4)
                    else:  # OTM
                        iv = np.random.uniform(0.3, 0.5)
                    
                    # Very simple price calculation
                    intrinsic = max(0, ticker_price - strike)
                    time_value = ticker_price * iv * np.sqrt(days_to_expiry / 365)
                    price = intrinsic + time_value
                    
                    calls_data.append({
                        'strike': strike,
                        'bid': price * 0.95,
                        'ask': price * 1.05,
                        'impliedVolatility': iv,
                        'volume': int(np.random.exponential(100) * (1.1 - abs(moneyness - 1))),
                        'openInterest': int(np.random.exponential(500) * (1.1 - abs(moneyness - 1)))
                    })
                
                # Create puts data
                puts_data = []
                for strike in strikes:
                    # Simple Black-Scholes approximation for IV
                    moneyness = strike / ticker_price
                    if moneyness > 1:  # ITM for puts
                        iv = np.random.uniform(0.25, 0.45)
                    else:  # OTM for puts
                        iv = np.random.uniform(0.35, 0.55)
                    
                    # Very simple price calculation
                    intrinsic = max(0, strike - ticker_price)
                    time_value = ticker_price * iv * np.sqrt(days_to_expiry / 365)
                    price = intrinsic + time_value
                    
                    puts_data.append({
                        'strike': strike,
                        'bid': price * 0.95,
                        'ask': price * 1.05,
                        'impliedVolatility': iv,
                        'volume': int(np.random.exponential(80) * (1.1 - abs(moneyness - 1))),
                        'openInterest': int(np.random.exponential(400) * (1.1 - abs(moneyness - 1)))
                    })
                
                # Convert to DataFrames
                ticker_options[exp_date] = {
                    'calls': pd.DataFrame(calls_data),
                    'puts': pd.DataFrame(puts_data)
                }
            
            options_data[ticker] = ticker_options
        
        return options_data
    
    def calculate_implied_volatility_surface(
        self,
        options_data: Dict,
        ticker: str,
        date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Calculates implied volatility surface from options data.
        
        Args:
            options_data: Dictionary with options data
            ticker: Ticker symbol
            date: Optional date to calculate for (latest if None)
            
        Returns:
            DataFrame with implied volatility surface
        """
        if ticker not in options_data:
            return pd.DataFrame()
        
        ticker_options = options_data[ticker]
        
        # Get all expiration dates
        expirations = list(ticker_options.keys())
        if not expirations:
            return pd.DataFrame()
        
        # Use latest date if not specified
        if date is None:
            date = max(expirations)
        elif date not in expirations:
            # Find nearest date
            nearest_date = min(expirations, key=lambda x: abs(pd.Timestamp(x) - pd.Timestamp(date)))
            date = nearest_date
        
        # Get options chain for this date
        chain = ticker_options[date]
        calls = chain.get('calls', pd.DataFrame())
        puts = chain.get('puts', pd.DataFrame())
        
        if calls.empty and puts.empty:
            return pd.DataFrame()
        
        # Combine call and put IVs
        iv_data = []
        
        # Process calls
        if not calls.empty and 'strike' in calls.columns and 'impliedVolatility' in calls.columns:
            for _, row in calls.iterrows():
                iv_data.append({
                    'strike': row['strike'],
                    'type': 'call',
                    'expiry': date,
                    'impliedVolatility': row['impliedVolatility']
                })
        
        # Process puts
        if not puts.empty and 'strike' in puts.columns and 'impliedVolatility' in puts.columns:
            for _, row in puts.iterrows():
                iv_data.append({
                    'strike': row['strike'],
                    'type': 'put',
                    'expiry': date,
                    'impliedVolatility': row['impliedVolatility']
                })
        
        # Convert to DataFrame
        iv_surface = pd.DataFrame(iv_data)
        
        return iv_surface
    
    def calculate_volatility_skew(self, iv_surface: pd.DataFrame) -> Dict:
        """
        Calculates volatility skew from IV surface.
        
        Args:
            iv_surface: DataFrame with implied volatility surface
            
        Returns:
            Dictionary with skew metrics
        """
        if iv_surface.empty:
            return {}
        
        # Calculate average IV for ATM options
        median_strike = iv_surface['strike'].median()
        atm_options = iv_surface[
            (iv_surface['strike'] >= median_strike * 0.95) &
            (iv_surface['strike'] <= median_strike * 1.05)
        ]
        
        if atm_options.empty:
            atm_iv = iv_surface['impliedVolatility'].mean()
        else:
            atm_iv = atm_options['impliedVolatility'].mean()
        
        # Calculate average IV for OTM puts (downside risk)
        otm_puts = iv_surface[
            (iv_surface['type'] == 'put') &
            (iv_surface['strike'] <= median_strike * 0.9)
        ]
        
        if otm_puts.empty:
            put_skew = 0
        else:
            otm_put_iv = otm_puts['impliedVolatility'].mean()
            put_skew = (otm_put_iv / atm_iv - 1) * 100  # Percentage skew
        
        # Calculate average IV for OTM calls (upside potential)
        otm_calls = iv_surface[
            (iv_surface['type'] == 'call') &
            (iv_surface['strike'] >= median_strike * 1.1)
        ]
        
        if otm_calls.empty:
            call_skew = 0
        else:
            otm_call_iv = otm_calls['impliedVolatility'].mean()
            call_skew = (otm_call_iv / atm_iv - 1) * 100  # Percentage skew
        
        # Calculate put-call skew spread
        skew_spread = put_skew - call_skew
        
        return {
            'atm_iv': atm_iv,
            'put_skew': put_skew,
            'call_skew': call_skew,
            'skew_spread': skew_spread
        }
    
    def calculate_term_structure(
        self,
        options_data: Dict,
        ticker: str
    ) -> pd.DataFrame:
        """
        Calculates volatility term structure across expirations.
        
        Args:
            options_data: Dictionary with options data
            ticker: Ticker symbol
            
        Returns:
            DataFrame with term structure
        """
        if ticker not in options_data:
            return pd.DataFrame()
        
        ticker_options = options_data[ticker]
        
        # Get all expiration dates
        expirations = list(ticker_options.keys())
        if not expirations:
            return pd.DataFrame()
        
        # Calculate ATM IV for each expiration
        term_structure = []
        
        for exp_date in expirations:
            chain = ticker_options[exp_date]
            calls = chain.get('calls', pd.DataFrame())
            puts = chain.get('puts', pd.DataFrame())
            
            if calls.empty and puts.empty:
                continue
            
            # Calculate days to expiry
            try:
                exp_dt = pd.Timestamp(exp_date)
                today = pd.Timestamp.now()
                days_to_expiry = (exp_dt - today).days
                if days_to_expiry < 0:
                    continue
            except:
                continue
            
            # Find the median strike (approximate ATM)
            all_strikes = []
            if not calls.empty and 'strike' in calls.columns:
                all_strikes.extend(calls['strike'].tolist())
            if not puts.empty and 'strike' in puts.columns:
                all_strikes.extend(puts['strike'].tolist())
            
            if not all_strikes:
                continue
                
            median_strike = np.median(all_strikes)
            
            # Find ATM options
            atm_calls = calls[
                (calls['strike'] >= median_strike * 0.95) &
                (calls['strike'] <= median_strike * 1.05)
            ] if not calls.empty else pd.DataFrame()
            
            atm_puts = puts[
                (puts['strike'] >= median_strike * 0.95) &
                (puts['strike'] <= median_strike * 1.05)
            ] if not puts.empty else pd.DataFrame()
            
            # Calculate average ATM IV
            atm_ivs = []
            if not atm_calls.empty and 'impliedVolatility' in atm_calls.columns:
                atm_ivs.extend(atm_calls['impliedVolatility'].tolist())
            if not atm_puts.empty and 'impliedVolatility' in atm_puts.columns:
                atm_ivs.extend(atm_puts['impliedVolatility'].tolist())
            
            if not atm_ivs:
                continue
                
            atm_iv = np.mean(atm_ivs)
            
            term_structure.append({
                'expiry': exp_date,
                'days_to_expiry': days_to_expiry,
                'atm_iv': atm_iv
            })
        
        # Convert to DataFrame
        term_df = pd.DataFrame(term_structure)
        
        # Sort by days to expiry
        if not term_df.empty:
            term_df = term_df.sort_values('days_to_expiry')
        
        return term_df
    
    def extract_options_signals(
        self,
        options_data: Dict,
        ticker: str
    ) -> Dict:
        """
        Extracts trading signals from options data.
        
        Args:
            options_data: Dictionary with options data
            ticker: Ticker symbol
            
        Returns:
            Dictionary with options-based signals
        """
        signals = {}
        
        # Calculate IV surface for the nearest expiry
        iv_surface = self.calculate_implied_volatility_surface(options_data, ticker)
        
        if not iv_surface.empty:
            # Calculate volatility skew
            skew_metrics = self.calculate_volatility_skew(iv_surface)
            signals.update(skew_metrics)
        
        # Calculate term structure
        term_structure = self.calculate_term_structure(options_data, ticker)
        
        if not term_structure.empty:
            # Calculate term structure slope (IV change per 30 days)
            if len(term_structure) >= 2:
                from scipy import stats
                
                # Linear regression on days vs IV
                slope, _, _, _, _ = stats.linregress(
                    term_structure['days_to_expiry'],
                    term_structure['atm_iv']
                )
                
                # Normalize to IV change per 30 days
                term_slope = slope * 30
                signals['term_structure_slope'] = term_slope
                
                # Calculate term structure shape (convexity)
                if len(term_structure) >= 3:
                    x = term_structure['days_to_expiry']
                    y = term_structure['atm_iv']
                    
                    # Fit quadratic polynomial
                    coeffs = np.polyfit(x, y, 2)
                    signals['term_structure_convexity'] = coeffs[0]  # Quadratic coefficient
        
        # Calculate IV percentile (compared to 1-year range)
        # This would require historical data, so we'll use a mock value
        signals['iv_percentile'] = np.random.uniform(0, 100)
        
        # Calculate put/call ratio
        # This would require volume data, so we'll use a mock value
        signals['put_call_ratio'] = np.random.uniform(0.5, 1.5)
        
        return signals
    
    def integrate_with_returns(
        self,
        returns: pd.DataFrame,
        options_signals: Dict[str, Dict],
        lookback: int = 20
    ) -> pd.DataFrame:
        """
        Integrates options signals into returns analysis.
        
        Args:
            returns: DataFrame with stock returns
            options_signals: Dictionary mapping tickers to options signals
            lookback: Lookback window for signals
            
        Returns:
            DataFrame with adjusted returns
        """
        # Copy returns to avoid modifying the original
        adjusted_returns = returns.copy()
        
        # Calculate return adjustments based on options signals
        for ticker in options_signals:
            if ticker not in adjusted_returns.columns:
                continue
            
            signals = options_signals[ticker]
            
            # Skip if no relevant signals
            if not signals:
                continue
            
            # Base adjustment on skew spread (higher skew spread -> more negative adjustment)
            skew_spread = signals.get('skew_spread', 0)
            
            # Normalize to a small adjustment factor (-0.1 to 0.1)
            adjustment = -skew_spread / 100  # Convert percentage to decimal
            adjustment = max(-0.1, min(0.1, adjustment))  # Limit range
            
            # Apply a small adjustment to recent returns
            last_n = min(lookback, len(adjusted_returns))
            adjusted_returns.loc[adjusted_returns.index[-last_n:], ticker] *= (1 + adjustment)
        
        return adjusted_returns

class OrderFlowProcessor:
    """
    Processes and integrates order flow data.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initializes the order flow processor.
        
        Args:
            api_key: Optional API key for data providers
        """
        self.api_key = api_key
    
    def fetch_order_flow_data(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        provider: str = 'mock'
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetches order flow data for the given tickers.
        
        Args:
            tickers: List of stock ticker symbols
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            provider: Data provider ('mock' only for now)
            
        Returns:
            Dictionary mapping tickers to DataFrames with order flow data
        """
        if provider == 'mock':
            return self._generate_mock_order_flow(tickers, start_date, end_date)
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def _generate_mock_order_flow(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str
    ) -> Dict[str, pd.DataFrame]:
        """
        Generates mock order flow data for demonstration.
        
        Args:
            tickers: List of stock ticker symbols
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            Dictionary mapping tickers to DataFrames with order flow data
        """
        # Create date range (minutely data)
        dates = pd.date_range(start=start_date, end=end_date, freq='1min')
        business_dates = [d for d in dates if d.weekday() < 5 and 9 <= d.hour < 16]
        
        # Initialize result
        result = {}
        
        for ticker in tickers:
            # Generate random stock price
            base_price = np.random.uniform(50, 500)
            
            # Generate price series with random walk
            price_changes = np.random.normal(0, 0.0005, len(business_dates))
            prices = base_price * np.cumprod(1 + price_changes)
            
            # Generate order flow data
            data = []
            
            for i, date in enumerate(business_dates):
                # Number of trades in this minute
                n_trades = np.random.poisson(20)
                
                # Current price
                current_price = prices[i]
                
                # Generate trades
                for _ in range(n_trades):
                    # Trade direction (-1 for sell, 1 for buy)
                    direction = np.random.choice([-1, 1], p=[0.48, 0.52])
                    
                    # Trade size
                    size = np.random.exponential(100)
                    if size > 1000:
                        size = 1000 + np.random.exponential(500)
                    size = int(size) + 1  # Ensure positive integer
                    
                    # Price impact
                    impact = direction * np.random.normal(0, 0.0002) * current_price
                    
                    # Trade price
                    price = current_price + impact
                    
                    # Market or limit order
                    order_type = np.random.choice(['market', 'limit'], p=[0.6, 0.4])
                    
                    data.append({
                        'timestamp': date,
                        'price': price,
                        'size': size,
                        'direction': direction,
                        'type': order_type
                    })
            
            # Convert to DataFrame
            result[ticker] = pd.DataFrame(data)
        
        return result
    
    def calculate_order_flow_imbalance(
        self,
        order_flow_data: pd.DataFrame,
        resample_freq: str = '1min'
    ) -> pd.Series:
        """
        Calculates order flow imbalance.
        
        Args:
            order_flow_data: DataFrame with order flow data
            resample_freq: Frequency for resampling
            
        Returns:
            Series with order flow imbalance
        """
        if order_flow_data.empty:
            return pd.Series()
        
        # Calculate signed volume (direction * size)
        order_flow_data['flow'] = order_flow_data['direction'] * order_flow_data['size']
        
        # Resample by time period
        resampled = order_flow_data.resample(resample_freq, on='timestamp')
        
        # Calculate net flow and total volume
        net_flow = resampled['flow'].sum()
        total_volume = resampled['size'].sum()
        
        # Calculate imbalance ratio
        imbalance = net_flow / total_volume
        
        return imbalance
    
    def calculate_vpin(
        self,
        order_flow_data: pd.DataFrame,
        bucket_size: Optional[int] = None,
        n_buckets: int = 50
    ) -> pd.Series:
        """
        Calculates Volume-synchronized Probability of Informed Trading (VPIN).
        
        Args:
            order_flow_data: DataFrame with order flow data
            bucket_size: Size of each volume bucket (or None to auto-calculate)
            n_buckets: Number of buckets to use for VPIN calculation
            
        Returns:
            Series with VPIN values
        """
        if order_flow_data.empty:
            return pd.Series()
        
        # Sort by timestamp
        data = order_flow_data.sort_values('timestamp')
        
        # Calculate total volume
        total_volume = data['size'].sum()
        
        # Auto-calculate bucket size if not provided
        if bucket_size is None:
            bucket_size = total_volume // (n_buckets * 5)  # Aim for 5x more buckets than needed for VPIN
        
        # Initialize buckets
        buy_volume = []
        sell_volume = []
        bucket_timestamps = []
        current_bucket_size = 0
        current_bucket_buy = 0
        current_bucket_sell = 0
        
        for _, row in data.iterrows():
            # Add to current bucket
            size = row['size']
            
            if row['direction'] > 0:  # Buy
                current_bucket_buy += size
            else:  # Sell
                current_bucket_sell += size
            
            current_bucket_size += size
            
            # Check if bucket is full
            if current_bucket_size >= bucket_size:
                buy_volume.append(current_bucket_buy)
                sell_volume.append(current_bucket_sell)
                bucket_timestamps.append(row['timestamp'])
                
                # Reset for next bucket
                current_bucket_size = 0
                current_bucket_buy = 0
                current_bucket_sell = 0
        
        # Add the last partial bucket if it has significant volume
        if current_bucket_size > bucket_size * 0.5:
            buy_volume.append(current_bucket_buy)
            sell_volume.append(current_bucket_sell)
            bucket_timestamps.append(data['timestamp'].iloc[-1])
        
        # Ensure we have enough buckets
        if len(buy_volume) < n_buckets:
            return pd.Series()
        
        # Calculate VPIN
        vpin_values = []
        vpin_timestamps = []
        
        for i in range(len(buy_volume) - n_buckets + 1):
            # Calculate VPIN over n_buckets
            bucket_group_buy = sum(buy_volume[i:i+n_buckets])
            bucket_group_sell = sum(sell_volume[i:i+n_buckets])
            total_bucket_volume = bucket_group_buy + bucket_group_sell
            
            if total_bucket_volume > 0:
                vpin = abs(bucket_group_buy - bucket_group_sell) / total_bucket_volume
                vpin_values.append(vpin)
                vpin_timestamps.append(bucket_timestamps[i+n_buckets-1])
        
        # Create result Series
        vpin_series = pd.Series(vpin_values, index=vpin_timestamps)
        
        return vpin_series
    
    def detect_large_orders(
        self,
        order_flow_data: pd.DataFrame,
        percentile_threshold: float = 99.0
    ) -> pd.DataFrame:
        """
        Detects abnormally large orders.
        
        Args:
            order_flow_data: DataFrame with order flow data
            percentile_threshold: Percentile threshold for "large" orders
            
        Returns:
            DataFrame with detected large orders
        """
        if order_flow_data.empty:
            return pd.DataFrame()
        
        # Calculate size threshold
        size_threshold = np.percentile(order_flow_data['size'], percentile_threshold)
        
        # Filter large orders
        large_orders = order_flow_data[order_flow_data['size'] >= size_threshold].copy()
        
        # Add a column indicating relative size
        large_orders['relative_size'] = large_orders['size'] / size_threshold
        
        return large_orders
    
    def integrate_order_flow_with_strategy(
        self,
        stock_returns: pd.DataFrame,
        order_flow_imbalance: Dict[str, pd.Series],
        lookback: int = 5
    ) -> Dict[str, float]:
        """
        Adjusts winner/loser classification using order flow data.
        
        Args:
            stock_returns: DataFrame with stock returns
            order_flow_imbalance: Dictionary mapping tickers to imbalance series
            lookback: Window for order flow analysis
            
        Returns:
            Dictionary mapping tickers to adjusted thresholds
        """
        # Initialize adjusted thresholds
        adjusted_thresholds = {}
        
        for ticker, returns in stock_returns.items():
            if ticker not in order_flow_imbalance:
                adjusted_thresholds[ticker] = 0.0  # Default value (no adjustment)
                continue
            
            # Get recent order flow imbalance
            imbalance = order_flow_imbalance[ticker]
            
            if imbalance.empty:
                adjusted_thresholds[ticker] = 0.0
                continue
            
            # Get most recent values
            recent_imbalance = imbalance.iloc[-lookback:].mean() if len(imbalance) >= lookback else imbalance.mean()
            
            # Calculate z-score if enough data
            if len(imbalance) > 20:
                imbalance_mean = imbalance.iloc[-20:].mean()
                imbalance_std = imbalance.iloc[-20:].std()
                if imbalance_std > 0:
                    z_score = (recent_imbalance - imbalance_mean) / imbalance_std
                else:
                    z_score = 0
            else:
                z_score = 0
            
            # Adjust threshold based on z-score
            # Strong buying pressure -> raise threshold for shorting (positive z-score)
            # Strong selling pressure -> raise threshold for longing (negative z-score)
            adjustment = 0.002 * z_score  # Scaling factor
            
            adjusted_thresholds[ticker] = adjustment
        
        return adjusted_thresholds

if __name__ == "__main__":
    # Example usage
    tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'PG']
    start_date = '2022-01-01'
    end_date = '2022-12-31'
    
    # Test sentiment data
    sentiment_processor = SentimentDataProcessor()
    news_sentiment = sentiment_processor.fetch_news_sentiment(
        tickers, start_date, end_date, provider='mock'
    )
    
    print("\nNews Sentiment Sample:")
    print(news_sentiment.head())
    
    # Test social sentiment data
    social_sentiment = sentiment_processor.fetch_social_sentiment(
        tickers, start_date, end_date, provider='mock'
    )
    
    print("\nSocial Sentiment Sample (Twitter):")
    for source, data in social_sentiment.items():
        print(f"\n{source.upper()} Sentiment:")
        print(data.head())
    
    # Aggregate sentiment
    aggregated = sentiment_processor.aggregate_sentiment(
        social_sentiment, method='exponential_decay'
    )
    
    print("\nAggregated Sentiment Sample:")
    print(aggregated.head())
    
    # Test options data
    options_processor = OptionsDataProcessor()
    options_data = options_processor.fetch_options_data(
        tickers[:3], start_date, end_date, provider='mock'
    )
    
    print("\nOptions Data Sample:")
    for ticker, data in options_data.items():
        print(f"\n{ticker} Options (first expiry):")
        first_expiry = list(data.keys())[0]
        print(f"Expiry: {first_expiry}")
        print("Calls:")
        print(data[first_expiry]['calls'].head())
        print("Puts:")
        print(data[first_expiry]['puts'].head())
        
        # Calculate IV surface
        iv_surface = options_processor.calculate_implied_volatility_surface(
            options_data, ticker
        )
        
        print(f"\n{ticker} IV Surface:")
        print(iv_surface.head())
        
        # Calculate skew
        skew = options_processor.calculate_volatility_skew(iv_surface)
        
        print(f"\n{ticker} Volatility Skew:")
        for metric, value in skew.items():
            print(f"{metric}: {value:.4f}")
        
        # Calculate term structure
        term_structure = options_processor.calculate_term_structure(
            options_data, ticker
        )
        
        print(f"\n{ticker} Term Structure:")
        print(term_structure.head())
        
        # Extract signals
        signals = options_processor.extract_options_signals(
            options_data, ticker
        )
        
        print(f"\n{ticker} Options Signals:")
        for signal, value in signals.items():
            print(f"{signal}: {value:.4f}")
    
    # Test order flow data
    flow_processor = OrderFlowProcessor()
    order_flow_data = flow_processor.fetch_order_flow_data(
        tickers[:2], start_date, end_date, provider='mock'
    )
    
    print("\nOrder Flow Data Sample:")
    for ticker, data in order_flow_data.items():
        print(f"\n{ticker} Order Flow:")
        print(data.head())
        
        # Calculate imbalance
        imbalance = flow_processor.calculate_order_flow_imbalance(
            data, resample_freq='1h'
        )
        
        print(f"\n{ticker} Order Flow Imbalance:")
        print(imbalance.head())
        
        # Calculate VPIN
        vpin = flow_processor.calculate_vpin(data)
        
        print(f"\n{ticker} VPIN:")
        print(vpin.head())
        
        # Detect large orders
        large_orders = flow_processor.detect_large_orders(data)
        
        print(f"\n{ticker} Large Orders:")
        print(large_orders.head())
