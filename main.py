import os
import pandas as pd
import numpy as np
import argparse
import time
from datetime import datetime
import matplotlib.pyplot as plt
import json

from data_preprocessing import prepare_stock_data, fetch_stock_data, calculate_returns, prepare_multifactor_data
from correlation_matrix import compute_correlation_matrix
from clustering import cluster_stocks, get_clusters_dict
from portfolio_construction import construct_and_evaluate_portfolios
from backtest import (
    StatisticalArbitrageBacktest, run_industry_benchmark,
    time_series_cross_validation, monte_carlo_backtest, walk_forward_optimization
)
from utils import (
    calculate_annualized_return,
    calculate_sharpe_ratio,
    calculate_sortino_ratio
)
# Use interactive Plotly visualizations
from visualization_plotly import (
    setup_visualization,
    plot_cumulative_returns,
    plot_performance_metrics,
    plot_cluster_analysis,
    plot_rolling_metrics,
    plot_drawdowns,
    plot_monthly_returns_heatmap,
    plot_transaction_costs,
    create_performance_report
)
from fama_french import get_fama_french_industries
from alternative_data import (
    SentimentDataProcessor,
    OptionsDataProcessor,
    OrderFlowProcessor
)
from adaptive_parameters import AdaptiveParameterManager
from portfolio_integration import PortfolioAllocator, calculate_information_ratio_impact
from config import (
    DATA_SETTINGS,
    CLUSTERING_SETTINGS,
    PORTFOLIO_SETTINGS,
    UNIVERSES,
    DEFAULT_UNIVERSE,
    VISUALIZATION_SETTINGS,
    OUTPUT_SETTINGS
)


def parse_arguments():
    """
    Parses command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Statistical Arbitrage Backtesting using Correlation Matrix Clustering'
    )

    # Data settings
    parser.add_argument('--start-date', type=str, default=DATA_SETTINGS['start_date'],
                        help='Start date for the backtest (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=DATA_SETTINGS['end_date'],
                        help='End date for the backtest (YYYY-MM-DD)')

    # Universe settings
    parser.add_argument('--universe', type=str, default=DEFAULT_UNIVERSE,
                        choices=list(UNIVERSES.keys()),
                        help='Stock universe to use')

    # Clustering settings
    parser.add_argument('--clustering-method', type=str,
                        default=CLUSTERING_SETTINGS['default_method'],
                        choices=CLUSTERING_SETTINGS['methods'],
                        help='Clustering method to use')

    parser.add_argument('--n-clusters-method', type=str,
                        default=CLUSTERING_SETTINGS['default_n_clusters_method'],
                        choices=CLUSTERING_SETTINGS['n_clusters_methods'],
                        help='Method to determine number of clusters')

    parser.add_argument('--fixed-n-clusters', type=int,
                        default=CLUSTERING_SETTINGS['fixed_n_clusters'],
                        help='Fixed number of clusters (if n-clusters-method is "fixed")')

    # Portfolio settings
    parser.add_argument('--lookback-window', type=int,
                        default=PORTFOLIO_SETTINGS['lookback_window'],
                        help='Lookback window for winner/loser identification (days)')

    parser.add_argument('--rebalance-period', type=int,
                        default=PORTFOLIO_SETTINGS['rebalance_period'],
                        help='Rebalance period (days)')

    parser.add_argument('--threshold', type=float,
                        default=PORTFOLIO_SETTINGS['threshold'],
                        help='Threshold for winner/loser classification')

    parser.add_argument('--stop-win', type=float,
                        default=PORTFOLIO_SETTINGS['stop_win_threshold'],
                        help='Stop-win threshold (set to 0 to disable)')

    # Output settings
    parser.add_argument('--output-dir', type=str, default=OUTPUT_SETTINGS['results_directory'],
                        help='Directory to save results')

    parser.add_argument('--report-name', type=str, default=None,
                        help='Base name for output report files')

    # Run settings
    parser.add_argument('--run-all-methods', action='store_true',
                        help='Run all clustering methods for comparison')

    parser.add_argument('--compare-to-industry', action='store_true',
                        help='Compare to industry-based strategy')

    parser.add_argument('--quick-test', action='store_true',
                        help='Run a quick test with fewer stocks and shorter time period')

    # Enhanced features
    parser.add_argument('--factor-model', type=str, default='capm',
                        choices=['capm', 'fama_french', 'pca'],
                        help='Factor model to use for residualization')

    parser.add_argument('--use-adaptive-params', action='store_true',
                        help='Use adaptive parameters based on market regimes')

    parser.add_argument('--use-alt-data', action='store_true',
                        help='Use alternative data sources for enhanced signal generation')

    parser.add_argument('--advanced-validation', action='store_true',
                        help='Run advanced validation (cross-validation, Monte Carlo, walk-forward)')

    parser.add_argument('--portfolio-integration', action='store_true',
                        help='Run portfolio integration analysis')

    return parser.parse_args()


def run_backtest(args):
    """
    Runs the statistical arbitrage backtest with the given arguments.

    Args:
        args: Parsed command line arguments

    Returns:
        Dictionary with backtest results
    """
    print(f"Starting backtest with the following parameters:")
    print(f"  Date range: {args.start_date} to {args.end_date}")
    print(f"  Universe: {args.universe}")
    print(f"  Clustering method: {args.clustering_method}")
    print(f"  N-clusters method: {args.n_clusters_method}")
    print(f"  Factor model: {args.factor_model}")
    print(f"  Using adaptive parameters: {args.use_adaptive_params}")
    print(f"  Using alternative data: {args.use_alt_data}")
    print()

    start_time = time.time()

    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup visualization
    setup_visualization()

    # Load universe of stocks
    universe_config = UNIVERSES[args.universe]
    if 'filepath' in universe_config:
        # Load tickers from file
        try:
            universe_df = pd.read_csv(universe_config['filepath'])
            tickers = universe_df['Ticker'].tolist()
        except Exception as e:
            print(f"Error loading universe file: {e}")
            tickers = []
    else:
        # Use predefined tickers
        tickers = universe_config['tickers']

    # For quick test, use fewer stocks
    if args.quick_test:
        print("Running quick test with reduced stock set and time period")
        tickers = tickers[:20]  # Use only top 20 stocks
        args.start_date = '2020-01-01'
        args.end_date = '2021-12-31'

    print(f"Using {len(tickers)} stocks from {args.universe} universe")

    # Get Fama-French industry classifications for benchmark
    industry_mapping = get_fama_french_industries(tickers)

    # Initialize result containers
    results = {
        'returns_data': {},
        'metrics_data': {},
        'cluster_histories': {},
        'adaptive_parameters': {},
        'alternative_data': {}
    }

    # Prepare data based on factor model
    print(f"\nPreparing data using {args.factor_model.upper()} factor model...")

    if args.factor_model == 'capm':
        # Use traditional CAPM approach
        returns, betas, residual_returns = prepare_stock_data(
            tickers,
            args.start_date,
            args.end_date,
            beta_window=60,
            market_cap_percentile=0.75
        )

        factor_returns = pd.DataFrame({'MKT': returns['SPY']})
        factor_data = {'betas': betas}
    else:
        # Use multi-factor model
        returns, factor_returns, residual_returns, factor_data = prepare_multifactor_data(
            tickers,
            args.start_date,
            args.end_date,
            factor_model=args.factor_model,
            rolling_window=60,
            market_cap_percentile=0.75
        )

    # Initialize adaptive parameter manager if requested
    adaptive_param_manager = None
    if args.use_adaptive_params:
        print("\nInitializing adaptive parameter manager...")
        adaptive_param_manager = AdaptiveParameterManager(
            returns,
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

        # Store adaptive parameters for some sample dates
        sample_dates = returns.index[
            np.linspace(0, len(returns.index) - 1, 10, dtype=int)
        ]

        adaptive_params = {}
        for date in sample_dates:
            params = adaptive_param_manager.get_regime_specific_parameters(date)
            adaptive_params[date.strftime('%Y-%m-%d')] = {
                k: float(v) if isinstance(v, (int, float)) else v
                for k, v in params.items()
            }

        results['adaptive_parameters'] = adaptive_params

    # Initialize alternative data if requested
    alt_data = {}
    if args.use_alt_data:
        print("\nGenerating alternative data...")

        # Initialize data processors
        sentiment_processor = SentimentDataProcessor()
        options_processor = OptionsDataProcessor()
        flow_processor = OrderFlowProcessor()

        # Generate mock data for demonstration

        # 1. Sentiment data
        print("  Generating sentiment data...")
        sentiment_data = sentiment_processor.fetch_news_sentiment(
            tickers[:20],  # Limit to 20 stocks for efficiency
            returns.index[-60].strftime('%Y-%m-%d'),
            returns.index[-1].strftime('%Y-%m-%d'),
            provider='mock'
        )

        # 2. Options data
        print("  Generating options data...")
        options_data = options_processor.fetch_options_data(
            tickers[:10],  # Limit to 10 stocks for efficiency
            returns.index[-60].strftime('%Y-%m-%d'),
            returns.index[-1].strftime('%Y-%m-%d'),
            provider='mock'
        )

        # Extract options signals
        options_signals = {}
        for ticker in tickers[:10]:
            options_signals[ticker] = options_processor.extract_options_signals(
                options_data, ticker
            )

        # 3. Order flow data
        print("  Generating order flow data...")
        order_flow_data = flow_processor.fetch_order_flow_data(
            tickers[:10],  # Limit to 10 stocks for efficiency
            returns.index[-60].strftime('%Y-%m-%d'),
            returns.index[-1].strftime('%Y-%m-%d'),
            provider='mock'
        )

        # Calculate flow imbalance
        flow_imbalance = {}
        for ticker, data in list(order_flow_data.items())[:5]:
            flow_imbalance[ticker] = flow_processor.calculate_order_flow_imbalance(
                data, resample_freq='1d'
            )

        # Store alternative data
        alt_data = {
            'sentiment': sentiment_data,
            'options_signals': options_signals,
            'flow_imbalance': flow_imbalance
        }

        results['alternative_data'] = {
            'available_data': list(alt_data.keys()),
            'tickers_covered': {
                'sentiment': list(sentiment_data.columns),
                'options_signals': list(options_signals.keys()),
                'flow_imbalance': list(flow_imbalance.keys())
            }
        }

    # Determine methods to run
    clustering_methods = CLUSTERING_SETTINGS['methods'] if args.run_all_methods else [args.clustering_method]

    # Set stop-win threshold (0 to disable)
    stop_win_threshold = args.stop_win if args.stop_win > 0 else None

    # Determine n_clusters parameter based on method
    if args.n_clusters_method == 'fixed':
        fixed_n_clusters = args.fixed_n_clusters
    else:
        fixed_n_clusters = None

    # Run backtest for each clustering method
    for method in clustering_methods:
        print(f"\nRunning backtest for {method} clustering...")

        # Create backtest instance
        backtest = StatisticalArbitrageBacktest(
            tickers=tickers,
            start_date=args.start_date,
            end_date=args.end_date,
            clustering_method=method,
            beta_window=60,
            correlation_window=20,
            lookback_window=args.lookback_window,
            rebalance_period=args.rebalance_period,
            threshold=args.threshold,
            stop_win_threshold=stop_win_threshold,
            market_cap_percentile=0.75,
            n_clusters_method=args.n_clusters_method,
            fixed_n_clusters=fixed_n_clusters,
            factor_model=args.factor_model,
            use_dynamic_parameters=args.use_adaptive_params,
            use_alternative_data=args.use_alt_data
        )

        # Set data directly if we already loaded it
        backtest.returns = returns
        backtest.residual_returns = residual_returns
        backtest.factor_returns = factor_returns
        backtest.factor_data = factor_data

        # Set adaptive parameter manager if available
        if adaptive_param_manager is not None:
            backtest.adaptive_param_manager = adaptive_param_manager

        # Add alternative data if available
        if alt_data:
            for data_type, data in alt_data.items():
                backtest.add_alternative_data(data_type, data)

        # Run the backtest
        portfolio_returns = backtest.run_backtest()
        metrics = backtest.get_performance_metrics()

        # Store results
        method_name = f"{method.replace('_', ' ').title()} Clustering"
        results['returns_data'][method_name] = portfolio_returns
        results['metrics_data'][method_name] = metrics
        results['cluster_histories'][method_name] = backtest.cluster_history

        # Print summary
        print(f"Results for {method_name}:")
        print(f"  Annualized Return: {metrics.loc['Combined', 'Annualized Return (%)']:.2f}%")
        print(f"  Sharpe Ratio: {metrics.loc['Combined', 'Sharpe Ratio']:.2f}")
        print(f"  Sortino Ratio: {metrics.loc['Combined', 'Sortino Ratio']:.2f}")
        print(f"  Maximum Drawdown: {metrics.loc['Combined', 'Maximum Drawdown (%)']:.2f}%")

        if 'Total Transaction Costs (%)' in metrics.loc['Combined'].index:
            print(f"  Transaction Costs: {metrics.loc['Combined', 'Total Transaction Costs (%)']:.2f}%")

    # Run industry benchmark if requested
    if args.compare_to_industry:
        print("\nRunning industry-based benchmark...")

        industry_returns, industry_metrics = run_industry_benchmark(
            tickers=tickers,
            start_date=args.start_date,
            end_date=args.end_date,
            industry_mapping=industry_mapping,
            beta_window=60,
            correlation_window=20,
            lookback_window=args.lookback_window,
            rebalance_period=args.rebalance_period,
            threshold=args.threshold,
            stop_win_threshold=stop_win_threshold
        )

        # Store results
        results['returns_data']['FF12 Industry'] = industry_returns
        results['metrics_data']['FF12 Industry'] = industry_metrics

        # Print summary
        print(f"Results for FF12 Industry benchmark:")
        print(f"  Annualized Return: {industry_metrics.loc['Combined', 'Annualized Return (%)']:.2f}%")
        print(f"  Sharpe Ratio: {industry_metrics.loc['Combined', 'Sharpe Ratio']:.2f}")
        print(f"  Sortino Ratio: {industry_metrics.loc['Combined', 'Sortino Ratio']:.2f}")
        print(f"  Maximum Drawdown: {industry_metrics.loc['Combined', 'Maximum Drawdown (%)']:.2f}%")

    # Get market benchmark returns
    print("\nFetching market benchmark data...")
    spy_prices = fetch_stock_data(
        ['SPY'],
        args.start_date,
        args.end_date,
        include_market=False
    )
    spy_returns = calculate_returns(spy_prices)['SPY']

    # Run advanced validation if requested
    if args.advanced_validation:
        print("\nRunning advanced validation...")

        # Time series cross-validation
        print("1. Time-series cross-validation...")
        tscv_results = time_series_cross_validation(
            tickers[:min(len(tickers), 20)],  # Limit tickers for speed
            args.start_date,
            args.end_date,
            n_splits=3,  # Limited splits for speed
            backtest_params={
                'clustering_method': args.clustering_method,
                'lookback_window': args.lookback_window,
                'rebalance_period': args.rebalance_period,
                'threshold': args.threshold,
                'factor_model': args.factor_model,
                'use_dynamic_parameters': args.use_adaptive_params,
                'use_alternative_data': args.use_alt_data
            }
        )

        results['cross_validation'] = {
            'mean_sharpe': float(tscv_results.get('mean_sharpe', 0)),
            'std_sharpe': float(tscv_results.get('std_sharpe', 0)),
            'mean_sortino': float(tscv_results.get('mean_sortino', 0)),
            'std_sortino': float(tscv_results.get('std_sortino', 0)),
            'mean_annualized_return': float(tscv_results.get('mean_annualized_return', 0)),
            'std_annualized_return': float(tscv_results.get('std_annualized_return', 0))
        }

        # Monte Carlo simulation
        print("2. Monte Carlo simulation...")
        mc_results = monte_carlo_backtest(
            returns.iloc[-252:],  # Last year for speed
            residual_returns.iloc[-252:],
            n_simulations=50,  # Limited sims for speed
            backtest_params={
                'clustering_method': args.clustering_method,
                'lookback_window': args.lookback_window,
                'rebalance_period': args.rebalance_period,
                'threshold': args.threshold
            },
            tickers=tickers[:min(len(tickers), 20)]  # Limit tickers for speed
        )

        results['monte_carlo'] = {
            'mean_sharpe': float(mc_results.get('mean', {}).get('sharpe_ratio', 0)),
            'std_sharpe': float(mc_results.get('std', {}).get('sharpe_ratio', 0)),
            'p5_sharpe': float(mc_results.get('5th_percentile', {}).get('sharpe_ratio', 0)),
            'p95_sharpe': float(mc_results.get('95th_percentile', {}).get('sharpe_ratio', 0)),
            'mean_return': float(mc_results.get('mean', {}).get('annualized_return', 0)),
            'std_return': float(mc_results.get('std', {}).get('annualized_return', 0))
        }

        # Walk-forward optimization
        print("3. Walk-forward optimization...")
        wf_results = walk_forward_optimization(
            tickers[:min(len(tickers), 20)],  # Limit tickers for speed
            args.start_date,
            args.end_date,
            initial_window=252,
            step=63,  # Quarterly steps
            param_grid={
                'lookback_window': [3, 5, 10],
                'rebalance_period': [1, 3, 5],
                'threshold': [0.0, 0.005]
            },
            clustering_method=args.clustering_method
        )

        results['walk_forward'] = {
            'weighted_sharpe': float(wf_results.get('weighted_sharpe', 0)),
            'weighted_sortino': float(wf_results.get('weighted_sortino', 0)),
            'weighted_return': float(wf_results.get('weighted_annual_return', 0)),
            'parameter_frequency': wf_results.get('parameter_frequency', {})
        }

    # Run portfolio integration analysis if requested
    if args.portfolio_integration:
        print("\nRunning portfolio integration analysis...")

        # Create mock strategy returns (for demonstration)
        main_strategy_returns = \
        results['returns_data'][f"{args.clustering_method.replace('_', ' ').title()} Clustering"]['Combined']

        # Market (SPY)
        market_returns = spy_returns.loc[spy_returns.index.isin(main_strategy_returns.index)]

        # Value strategy (simple simulation)
        value_returns = pd.Series(
            [0.0004 + np.sin(i / 180) * 0.001 + np.random.normal(0, 0.01) for i in range(len(main_strategy_returns))],
            index=main_strategy_returns.index
        )

        # Momentum strategy (simple simulation)
        momentum_returns = pd.Series(
            [0.0005 + (0.1 if i > 0 and value_returns.iloc[i - 1] > 0 else -0.1) * 0.001 + np.random.normal(0, 0.012)
             for i in range(len(main_strategy_returns))],
            index=main_strategy_returns.index
        )

        # Create strategies dictionary
        strategies = {
            'Market': market_returns,
            'Value': value_returns,
            'Momentum': momentum_returns,
            'StatArb': main_strategy_returns
        }

        # Initialize portfolio allocator
        allocator = PortfolioAllocator(strategies, risk_free_rate=0.005)

        # Calculate optimal allocations
        allocation_results = {}

        for method in ['risk_parity', 'min_variance', 'max_diversification', 'equal_weight']:
            weights = allocator.optimize_allocation(method)
            metrics = allocator.calculate_portfolio_metrics(weights)

            allocation_results[method] = {
                'weights': weights.to_dict(),
                'metrics': {
                    'annualized_return': metrics['Annualized Return'],
                    'annualized_volatility': metrics['Annualized Volatility'],
                    'sharpe_ratio': metrics['Sharpe Ratio'],
                    'max_drawdown': metrics['Max Drawdown'],
                    'diversification_ratio': metrics.get('Diversification Ratio', 0)
                }
            }

        # Get strategy clusters
        strategy_clustering = allocator.cluster_strategies(threshold=0.5)

        # Calculate incremental benefit of StatArb to Market portfolio
        impact = calculate_information_ratio_impact(
            market_returns,
            main_strategy_returns,
            allocation=0.15
        )

        # Store portfolio integration results
        results['portfolio_integration'] = {
            'strategy_correlations': allocator.correlation_matrix.to_dict(),
            'strategy_metrics': {
                name: {
                    'annualized_return': float(row['Annualized Return']),
                    'annualized_volatility': float(row['Annualized Volatility']),
                    'sharpe_ratio': float(row['Sharpe Ratio']),
                    'max_drawdown': float(row['Max Drawdown']),
                }
                for name, row in allocator.metrics.iterrows()
            },
            'allocations': allocation_results,
            'strategy_clusters': {
                k: v for k, v in strategy_clustering['clusters'].items()
            },
            'incremental_impact': {
                k: float(v) for k, v in impact.items()
            }
        }

    # Generate visualizations
    print("\nGenerating visualizations...")

    # Determine report name
    if args.report_name:
        report_base = args.report_name
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_base = f"stat_arb_{args.universe}_{args.clustering_method}_{timestamp}"

    # Plot cumulative returns
    plot_cumulative_returns(
        results['returns_data'],
        spy_returns,
        title='Cumulative Returns Comparison',
        filename=f"{report_base}_cumulative_returns.{VISUALIZATION_SETTINGS['plot_format']}"
    )

    # Plot metrics comparisons
    for metric in ['Annualized Return (%)', 'Sharpe Ratio', 'Sortino Ratio']:
        plot_performance_metrics(
            results['metrics_data'],
            metric,
            title=f'{metric} Comparison',
            filename=f"{report_base}_{metric.lower().replace(' ', '_').replace('(%)', 'pct')}_comparison.{VISUALIZATION_SETTINGS['plot_format']}"
        )

    # Plot drawdowns
    plot_drawdowns(
        results['returns_data'],
        title='Strategy Drawdowns',
        filename=f"{report_base}_drawdowns.{VISUALIZATION_SETTINGS['plot_format']}"
    )

    # Plot transaction costs if available
    for method_name, returns_df in results['returns_data'].items():
        if any(col.endswith('_TC') for col in returns_df.columns):
            plot_transaction_costs(
                returns_df,
                title=f'Transaction Costs: {method_name}',
                filename=f"{report_base}_{method_name.lower().replace(' ', '_')}_transaction_costs.{VISUALIZATION_SETTINGS['plot_format']}"
            )

    # Plot rolling metrics for the main method
    main_method = f"{args.clustering_method.replace('_', ' ').title()} Clustering"
    if main_method in results['returns_data']:
        plot_rolling_metrics(
            {main_method: results['returns_data'][main_method]},
            window=252,
            metric='sharpe',
            title=f'Rolling 1-Year Sharpe Ratio: {main_method}',
            filename=f"{report_base}_rolling_sharpe.{VISUALIZATION_SETTINGS['plot_format']}"
        )

        plot_monthly_returns_heatmap(
            results['returns_data'],
            main_method,
            title=f'Monthly Returns: {main_method}',
            filename=f"{report_base}_monthly_returns.{VISUALIZATION_SETTINGS['plot_format']}"
        )

    # Plot cluster analysis for the main method
    if main_method in results['cluster_histories'] and results['cluster_histories'][main_method]:
        plot_cluster_analysis(
            results['cluster_histories'][main_method],
            title=f'Number of Clusters Over Time: {main_method}',
            filename=f"{report_base}_cluster_analysis.{VISUALIZATION_SETTINGS['plot_format']}"
        )

    # Create performance report
    report_path = os.path.join(args.output_dir, f"{report_base}_report.md")
    create_performance_report(
        results['returns_data'],
        results['metrics_data'],
        spy_returns,
        output_file=report_path
    )

    # Save results as JSON for further analysis
    try:
        # Convert results to JSON-serializable format
        serializable_results = {}

        # Save basic run parameters
        serializable_results['parameters'] = {
            'start_date': args.start_date,
            'end_date': args.end_date,
            'universe': args.universe,
            'clustering_method': args.clustering_method,
            'n_clusters_method': args.n_clusters_method,
            'lookback_window': args.lookback_window,
            'rebalance_period': args.rebalance_period,
            'threshold': args.threshold,
            'stop_win': args.stop_win,
            'factor_model': args.factor_model,
            'use_adaptive_params': args.use_adaptive_params,
            'use_alt_data': args.use_alt_data
        }

        # Save metrics
        serializable_results['metrics'] = {
            method: metrics.loc['Combined'].to_dict()
            for method, metrics in results['metrics_data'].items()
        }

        # Save validation results if available
        for key in ['cross_validation', 'monte_carlo', 'walk_forward', 'portfolio_integration']:
            if key in results:
                serializable_results[key] = results[key]

        # Save results
        results_path = os.path.join(args.output_dir, f"{report_base}_results.json")
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        print(f"Results saved to {results_path}")
    except Exception as e:
        print(f"Error saving results as JSON: {e}")

    # Calculate execution time
    execution_time = time.time() - start_time
    print(f"\nBacktest completed in {execution_time:.2f} seconds")
    print(f"Results saved to {args.output_dir}")

    return results


def main():
    """
    Main entry point for the program.
    """
    args = parse_arguments()
    results = run_backtest(args)


if __name__ == "__main__":
    main()