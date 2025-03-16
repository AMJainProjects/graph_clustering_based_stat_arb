# Correlation Matrix Clustering for Statistical Arbitrage

This repository implements the methods described in the paper "Correlation Matrix Clustering for Statistical Arbitrage Portfolios" by Cartea, Cucuringu, and Jin (2023), with significant enhancements to the original methodology.

## Overview

Statistical arbitrage strategies aim to exploit temporary price deviations among similar assets. This implementation focuses on a two-step approach:

1. **Group similar assets**: Use graph clustering algorithms to partition stocks into groups where stocks within the same group tend to co-move.
2. **Create arbitrage portfolios**: Within each cluster, construct mean-reverting portfolios that exploit temporary deviations from the cluster's average return.

## Mathematical Foundation

### 1. Residual Returns

We start by calculating market residual returns for each stock, removing the market factor influence:

$$R^{res}_{i,t} = R_{i,t} - \beta_i R_{mkt,t}$$

Where:
- $R_{i,t}$ is the raw return of stock $i$ at time $t$
- $\beta_i$ is the stock's sensitivity to market movements (CAPM beta)
- $R_{mkt,t}$ is the market return at time $t$ (S&P 500/SPY ETF)

**Why this approach?** The one-factor CAPM adjustment aims to isolate stock-specific movements by removing market-wide effects. This is crucial because we want to identify relative value opportunities within sectors/clusters rather than taking directional market bets. By removing the market factor, we reduce the risk of our strategy being merely a disguised bet on market direction.

### 2. Advanced Factor Models (Enhanced)

While CAPM provides a basic market adjustment, research has shown that multiple factors drive stock returns. We extend beyond the simple one-factor model to more sophisticated approaches:

#### 2.1 Multi-Factor Models

Generalized residualization process using multiple factors:

$$R^{res}_{i,t} = R_{i,t} - \sum_{k=1}^{K} \beta_{i,k} F_{k,t}$$

Where:
- $F_{k,t}$ is the return of factor $k$ at time $t$
- $\beta_{i,k}$ is the sensitivity of stock $i$ to factor $k$

Implemented factor models include:
- **Fama-French Factors**: Using the 5-factor model (Market, Size, Value, Profitability, Investment)
- **PCA-Based Factors**: Data-driven factors extracted from the return matrix

**Why enhance with multi-factor models?** Single-factor CAPM fails to account for well-documented risk premia like size, value, and momentum effects. By extending to multi-factor models:

1. **Improved signal isolation**: We remove more systematic effects, leading to cleaner stock-specific signals
2. **Reduced factor exposure**: The strategy becomes more market-neutral across multiple dimensions
3. **Better clustering accuracy**: With more systematic effects removed, correlations better reflect true stock relationships
4. **Adaptability to changing markets**: Data-driven PCA factors can identify emerging drivers of returns that traditional models might miss

The PCA approach is particularly valuable as it extracts factors directly from the data without relying on predefined economic interpretations, potentially capturing market dynamics that established models might miss.

### 3. Correlation Matrix

We compute the correlation matrix of these residual returns:

$$C_{i,j} = \frac{\sum_{t=T-w}^{T-1}(R^{res}_{t,i} - \bar{R}^{res}_i)(R^{res}_{t,j} - \bar{R}^{res}_j)}{(w-1)\sigma_i\sigma_j}$$

Where:
- $\bar{R}^{res}_i$ is the mean residual return of stock $i$
- $\sigma_i$ and $\sigma_j$ are standard deviations
- $w$ is the lookback window size

**Why correlation matrices?** Correlation matrices provide a natural measure of co-movement between assets. Using residual returns rather than raw returns ensures that these correlations capture relationships beyond common market exposure. The correlation structure serves as the foundation for our clustering approach, helping identify natural groupings of stocks that tend to move together after accounting for systematic factors.

### 4. Clustering Algorithms

The implementation includes multiple clustering methods:

#### 4.1 Basic Clustering Methods

- **Spectral Clustering**: Uses the spectrum of the graph Laplacian matrix. Since the correlation matrix may contain negative entries, we use the absolute values.

  The Laplacian matrix is defined as:
  $$L = D - A$$

  Where:
  - $A$ is the adjacency matrix (absolute correlation matrix)
  - $D$ is the diagonal degree matrix with $D_{ii} = \sum_{j=1}^{n} A_{ij}$

- **Signed Laplacian (Random Walk)**:  
  This handles signed networks directly, using:
  $$\bar{L}_{rw} = I - \bar{D}^{-1}A$$

  Where $\bar{D}_{ii} = \sum_{j=1}^{n} |A_{ij}|$

- **Signed Laplacian (Symmetric)**:  
  A symmetric normalized variant:
  $$\bar{L}_{sym} = I - \bar{D}^{-1/2}A\bar{D}^{-1/2}$$

- **SPONGE**:  
  SPONGE (Signed Positive Over Negative Generalized Eigenproblem) decomposes the adjacency matrix into positive and negative components:
  $$A = A^+ - A^-$$

  And solves a generalized eigenvalue problem:
  $$(L^+ + \tau^- D^-, L^- + \tau^+ D^+)$$

- **SPONGEsym**:  
  A symmetric variant of SPONGE using the symmetric Laplacian.

**Why these clustering approaches?** Financial correlation matrices present unique challenges:

1. **Signed graphs**: Unlike many network problems, financial correlations can be negative (assets moving in opposite directions), requiring specialized methods like Signed Laplacian and SPONGE
2. **Different information in positive and negative correlations**: Negative correlations may indicate hedging relationships just as important as positive co-movement
3. **Noise sensitivity**: Financial data contains substantial noise, requiring robust algorithms
4. **Non-spherical clusters**: Market sectors often have complex shapes that aren't easily captured by simpler clustering methods

SPONGE methods in particular are valuable because they handle both positive and negative correlations in a principled way, balancing attractive forces (positive correlations) and repulsive forces (negative correlations) to find optimal clusters.

#### 4.2 Advanced Clustering Methods (Enhanced)

- **Hierarchical Clustering with Dynamic Tree Cutting**: Builds a dendrogram and intelligently cuts it to identify natural groupings.
- **Density-Based Clustering (HDBSCAN)**: Identifies clusters of varying densities and handles noise points.
- **Deep Embedding Clustering**: Uses autoencoder neural networks to learn low-dimensional embeddings for clustering.

**Why enhance with these methods?** Each addresses specific limitations of traditional approaches:

1. **Hierarchical Clustering with Dynamic Tree Cutting**: 
   - **Problem solved**: Fixed number of clusters may not reflect natural market structure
   - **Benefits**: Adaptively determines cluster boundaries based on actual data structure
   - **Advantage**: Creates interpretable hierarchy of relationships, useful for risk monitoring

2. **Density-Based Clustering (HDBSCAN)**:
   - **Problem solved**: Standard methods struggle with varying cluster densities and outliers
   - **Benefits**: Identifies stocks with unique behavior as outliers rather than forcing them into clusters
   - **Advantage**: More robust to market disruptions where correlation structures temporarily break down

3. **Deep Embedding Clustering**:
   - **Problem solved**: Linear methods may miss complex non-linear relationships
   - **Benefits**: Captures subtle patterns in return dynamics beyond simple correlations
   - **Advantage**: Can adapt to changing market conditions by learning relevant features directly from data

Together, these enhanced methods provide more flexible, adaptive clustering that better respects the natural structure in financial data.

### 5. Determining Number of Clusters

We implement two methods:

#### 5.1 Marchenko-Pastur Distribution

We select eigenvalues exceeding the upper boundary of the MP distribution:
$$\lambda_+ = (1 + \sqrt{\rho})^2$$
where $\rho = N/T$ is the ratio of stocks to time periods.

#### 5.2 Variance Explained

Select the top-k eigenvalues that explain a specified percentage of total variance:
$$\frac{\sum_{i=1}^{k} \lambda_i}{\sum_{i=1}^{N} \lambda_i} \geq P$$

**Why these approaches?** Determining the optimal number of clusters is a critical challenge:

1. **Marchenko-Pastur (MP) method**:
   - **Scientific foundation**: Based on random matrix theory (RMT), which provides a sound theoretical framework
   - **Signal vs. noise separation**: Distinguishes statistically significant correlations from random noise
   - **Adaptive**: Accounts for the ratio of assets to time periods, crucial in high-dimensional financial data

2. **Variance Explained method**:
   - **Interpretability**: Directly relates to how much of the return variation is captured
   - **Practical tuning**: Allows practitioners to balance complexity vs. explanatory power
   - **Stability**: Tends to produce more stable clusters over time

Using both methods provides complementary perspectives - the MP approach offers theoretical rigor while variance explained provides practical flexibility.

### 6. Portfolio Construction

Within each cluster, we:
1. Calculate the mean return of all stocks
2. Identify "winners" (outperformers) and "losers" (underperformers)
3. Take a contrarian position: short winners, long losers
4. Normalize weights to create a zero-cost portfolio

**Why this strategy?** The portfolio construction follows a statistical arbitrage logic:

1. **Mean-reversion exploitation**: Research shows stock returns tend to exhibit short-term reversal within similar asset groups
2. **Diversification benefit**: By constructing multiple independent portfolios across clusters, we reduce overall strategy risk
3. **Market neutrality**: Zero-cost construction within clusters helps maintain neutrality to broad market movements
4. **Sector neutrality**: By operating within clusters rather than across the entire market, we avoid inadvertent sector bets

This approach harnesses both statistical principles (mean reversion) and financial market structure (clusters representing economic relationships) to identify temporary mispricings.

### 7. Dynamic Parameter Optimization (Enhanced)

We implement regime-switching models to adaptively adjust strategy parameters:

$$\theta_t = f(S_t)$$

Where:
- $\theta_t$ is the set of strategy parameters at time $t$
- $S_t$ is the detected market regime state
- $f$ is a mapping function from regimes to optimal parameters

Key adaptive parameters include:
- Lookback window length
- Rebalancing frequency
- Winner/loser classification threshold
- Correlation window size

**Why dynamic parameters?** Markets operate in distinct regimes with different characteristics:

1. **Market regime sensitivity**: 
   - **Problem solved**: Fixed parameters that work in calm markets often fail during high volatility periods
   - **Approach**: Hidden Markov Models (HMMs) identify distinct volatility regimes
   - **Benefit**: Parameters automatically adjust to prevailing conditions

2. **Adaptivity to changing market speeds**:
   - **Problem solved**: Market reaction speeds vary across regimes (high volatility periods typically move faster)
   - **Implementation**: Shorter lookback windows during volatile periods, longer during calm markets
   - **Advantage**: Captures the appropriate signal horizon for current conditions

3. **Threshold optimization**:
   - **Problem solved**: Fixed thresholds may be too strict during low-volatility periods and too loose during high volatility
   - **Method**: Dynamically scale thresholds based on recent volatility
   - **Result**: More consistent portfolio turnover across different market environments

This adaptive approach helps avoid the pitfall of parameters that work well in backtest but fail when market conditions change.

### 8. Alternative Data Integration (Enhanced)

We incorporate non-price information sources to enhance clustering and signal generation:

#### 8.1 Sentiment Analysis

Sentiment scores are used to adjust expected returns:

$$\tilde{R}_{i,t} = R_{i,t} \cdot (1 + \alpha \cdot sentiment_{i,t})$$

Where $\alpha$ is a scaling factor for sentiment impact.

**Why sentiment data?**
- **Forward-looking information**: Captures market expectations not yet reflected in price
- **Crowdsourced intelligence**: Aggregates views across many market participants
- **Lead indicator**: Often precedes price movements, especially around significant events
- **Implementation benefit**: More subtle than price movements, potentially identifying opportunities earlier

#### 8.2 Options-Derived Signals

Implied volatility skew and term structure provide forward-looking information:

$$threshold_{i,t} = base\_threshold + \beta \cdot skew\_signal_{i,t}$$

**Why options data?**
- **Market expectations**: Contains implied forecasts of future volatility and price direction
- **Institutional insight**: Often reflects positioning of sophisticated investors
- **Risk premium indicator**: Captures market pricing of tail risks
- **Advantage**: Provides a market-derived probability distribution of future returns, not just point estimates

#### 8.3 Order Flow Imbalance

Market microstructure information used to adjust thresholds:

$$threshold_{i,t} = base\_threshold + \gamma \cdot flow\_imbalance_{i,t}$$

**Why order flow data?**
- **Real-time adaptation**: Captures current market behavior at the microstructure level
- **Supply-demand imbalances**: Identifies temporary liquidity issues versus fundamental repricing
- **Informed trading detection**: May help distinguish noise from information-driven price movements
- **Implementation value**: Particularly helpful for avoiding trades against strong order flow momentum

By integrating these alternative data sources, the strategy gains multiple perspectives beyond just historical prices, potentially identifying opportunities earlier and with greater conviction.

## Implementation Details

The codebase is organized as follows:

### Core Components
- `data_preprocessing.py`: Handles data loading, cleaning, beta calculation, and residual returns
- `correlation_matrix.py`: Builds correlation matrices from residual returns
- `clustering.py`: Implements all clustering algorithms
- `portfolio_construction.py`: Creates and rebalances portfolios based on clustering
- `backtest.py`: Evaluates strategy performance
- `utils.py`: Utility functions for metrics and visualizations
- `main.py`: End-to-end execution

### Enhanced Components
- `adaptive_parameters.py`: Implements dynamic parameter optimization based on market regimes
- `alternative_data.py`: Processes and integrates sentiment, options, and order flow data
- `portfolio_integration.py`: Manages integration with existing portfolios and capital allocation
- `visualization_plotly.py`: Interactive visualizations using Plotly

## Backtesting Framework

### Core Backtesting
The backtesting framework:
1. Prepares historical price data
2. Calculates residual returns
3. Periodically rebalances portfolios based on clustering
4. Tracks transaction costs
5. Evaluates performance metrics

### Backtesting Extensions (Enhanced)

The framework includes advanced backtesting techniques:

#### Out-of-Sample Validation
Time-series cross-validation is used to assess strategy robustness across different market regimes:
```python
validation_results = time_series_cross_validation(
    tickers, start_date, end_date, 
    n_splits=5, 
    backtest_params={'clustering_method': 'sponge_sym'}
)
```

**Why time-series cross-validation?**
- **Problem addressed**: Simple train-test splits don't account for regime changes and time-varying relationships
- **Method**: Multiple forward-testing periods with expanding windows preserve temporal order
- **Benefit**: More realistic assessment of how the strategy would have performed in real trading
- **Advantage over alternatives**: Produces distribution of results across different market periods, revealing strategy robustness

#### Monte Carlo Simulation
Bootstrapping techniques are employed to generate confidence intervals for performance metrics:
```python
mc_results = monte_carlo_backtest(
    returns, residual_returns, 
    n_simulations=1000,
    bootstrap_method='block'  # Block bootstrap preserves autocorrelation
)
```

**Why Monte Carlo simulation?**
- **Problem addressed**: Historical results represent just one realization from a distribution of possibilities
- **Method**: Block bootstrap resampling preserves return autocorrelation structure while generating alternative scenarios
- **Benefit**: Produces confidence intervals around performance metrics rather than point estimates
- **Key insight gained**: Reveals strategy's sensitivity to specific historical sequences versus robust patterns

#### Walk-Forward Optimization
Parameters are continually re-optimized as new data arrives to maintain adaptivity:
```python
wfo_results = walk_forward_optimization(
    tickers, start_date, end_date,
    initial_window=252,  # 1 year initial window
    step=20,             # Re-optimize every 20 days
    param_grid={
        'lookback_window': [3, 5, 10, 15],
        'rebalance_period': [1, 3, 5]
    }
)
```

**Why walk-forward optimization?**
- **Problem addressed**: Static parameters optimized on full sample lead to look-ahead bias and parameter overfitting
- **Method**: Parameters periodically retrained on expanding window, used only on unseen data
- **Benefit**: Mimics real-world process of periodic strategy recalibration
- **Implementation advantage**: Provides insights on parameter stability across different market regimes

These advanced validation techniques provide a more comprehensive understanding of strategy performance and robustness than simple full-sample backtests.

## Portfolio Integration (Enhanced)

We provide methods to integrate the statistical arbitrage strategy with existing portfolios:

### Optimal Allocation Methods
- **Risk Parity**: Equal risk contribution across strategies
- **Minimum Variance**: Minimizes portfolio volatility
- **Maximum Diversification**: Maximizes the diversification ratio
- **Risk Budgeting**: Allocates according to specified risk targets

**Why these allocation methods?**
- **Problem addressed**: Naïve equal weighting doesn't account for different strategy risk profiles and correlations
- **Risk Parity approach**: Allocates capital to equalize risk contribution, preventing any single strategy from dominating risk
- **Minimum Variance benefit**: Finding the mix that minimizes overall volatility
- **Maximum Diversification advantage**: Optimizes for diversification potential rather than just risk reduction
- **Risk Budgeting value**: Allows flexible capital allocation aligned with risk tolerance

### Strategy Clustering
Hierarchical clustering is used to identify genuinely diversifying strategies:
```python
allocator = PortfolioAllocator(strategies)
clustering = allocator.cluster_strategies(threshold=0.5)
print(f"Identified {clustering['n_clusters']} strategy clusters")
```

**Why cluster strategies?**
- **Problem addressed**: Superficially different strategies may actually capture the same underlying factor
- **Method**: Hierarchical clustering identifies natural groupings of strategies based on return correlations
- **Benefit**: Ensures capital is allocated to truly diverse alpha sources
- **Implementation advantage**: Helps identify redundant strategies that can be eliminated or re-weighted

### Incremental Benefit Analysis
Measures the impact of adding statistical arbitrage to an existing portfolio:
```python
impact = calculate_information_ratio_impact(
    existing_portfolio_returns,
    stat_arb_returns,
    allocation=0.1  # 10% allocation to stat arb
)
print(f"Sharpe improvement: {impact['sharpe_improvement']:.2f}")
```

**Why incremental benefit analysis?**
- **Problem addressed**: Raw strategy metrics don't show marginal benefit to an existing portfolio
- **Method**: Information ratio measures excess return per unit of tracking error
- **Benefit**: Quantifies the value added by including the new strategy
- **Key insight gained**: A strategy with modest standalone metrics may still provide substantial portfolio benefit if its returns are uncorrelated with existing holdings

These portfolio integration tools help practitioners make informed allocation decisions and understand the true value added by the statistical arbitrage strategy.

## Performance Metrics

We evaluate performance using:

1. **Annualized Return**: Average yearly return of the strategy
2. **Sharpe Ratio**: Risk-adjusted return using standard deviation as risk
   $$\text{Sharpe Ratio} = \frac{\text{Portfolio Return}}{\text{Standard Deviation}}$$
3. **Sortino Ratio**: Risk-adjusted return focusing on downside risk
   $$\text{Sortino Ratio} = \frac{\text{Portfolio Return}}{\text{Downside Deviation}}$$
4. **Information Ratio**: Excess return relative to a benchmark per unit of tracking error
   $$\text{Information Ratio} = \frac{\text{Strategy Return} - \text{Benchmark Return}}{\text{Tracking Error}}$$

## Requirements

- Python 3.7+
- NumPy
- pandas
- SciPy
- scikit-learn
- matplotlib
- plotly
- yfinance (for data fetching)
- tensorflow (for deep clustering)
- hdbscan (for density-based clustering)

## Usage

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the main script: `python main.py`

## Advanced Usage

```python
# Basic backtest with default parameters
backtest = StatisticalArbitrageBacktest(
    tickers=tickers,
    start_date=start_date,
    end_date=end_date
)
backtest.run_backtest()

# Advanced backtest with dynamic parameters and alternative data
backtest = StatisticalArbitrageBacktest(
    tickers=tickers,
    start_date=start_date,
    end_date=end_date,
    factor_model='fama_french',
    use_dynamic_parameters=True,
    use_alternative_data=True
)

# Add sentiment data
sentiment_processor = SentimentDataProcessor()
sentiment_data = sentiment_processor.fetch_news_sentiment(tickers, start_date, end_date)
backtest.add_alternative_data('sentiment', sentiment_data)

# Run backtest
returns = backtest.run_backtest()
metrics = backtest.get_performance_metrics()

# Portfolio integration
allocator = PortfolioAllocator({
    'StatArb': returns['Combined'],
    'Market': market_returns,
    'Value': value_returns
})
weights = allocator.optimize_allocation(method='risk_parity')
```

## Configuration

You can adjust strategy parameters in `config.py`:

- Lookback window for beta calculation
- Rebalancing frequency
- Number of clusters or method for determining it
- Threshold for identifying winners/losers
- Stop-win threshold
- Factor model specifications
- Alternative data integration settings

## References

1. Cartea, Á., Cucuringu, M., & Jin, Q. (2023). Correlation Matrix Clustering for Statistical Arbitrage Portfolios.
2. Fama, E. F., & French, K. R. (2015). A five-factor asset pricing model. Journal of Financial Economics, 116(1), 1-22.
3. Campello, R. J., Moulavi, D., & Sander, J. (2013). Density-based clustering based on hierarchical density estimates. Pacific-Asia conference on knowledge discovery and data mining.
4. Langfelder, P., Zhang, B., & Horvath, S. (2008). Defining clusters from a hierarchical cluster tree: the Dynamic Tree Cut package for R. Bioinformatics, 24(5), 719-720.
5. Patton, A. J., & Timmermann, A. (2012). Portfolio sorts and tests of cross-sectional patterns in expected returns. Journal of Financial Economics, 104(1), 141-161.
6. Timmermann, A. (2018). Forecasting methods in finance. Annual Review of Financial Economics, 10, 449-479.
7. Plerou, V., Gopikrishnan, P., Rosenow, B., Amaral, L. A. N., & Stanley, H. E. (2000). Random matrix approach to cross correlations in financial data. Physical Review E, 65(6), 066126.
8. Bouchaud, J. P., & Potters, M. (2003). Theory of financial risk and derivative pricing: from statistical physics to risk management. Cambridge University Press.