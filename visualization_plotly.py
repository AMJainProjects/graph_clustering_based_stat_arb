import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional, Union
import os
from datetime import datetime

from config import VISUALIZATION_SETTINGS, OUTPUT_SETTINGS

def setup_visualization():
    """
    Sets up the visualization environment.
    """
    # Create plots directory if it doesn't exist
    if VISUALIZATION_SETTINGS['save_plots'] and not os.path.exists(OUTPUT_SETTINGS['plots_directory']):
        os.makedirs(OUTPUT_SETTINGS['plots_directory'])

def plot_cumulative_returns(
    returns_data: Dict[str, pd.DataFrame],
    benchmark_returns: Optional[pd.Series] = None,
    title: str = 'Cumulative Returns',
    filename: Optional[str] = None
):
    """
    Plots cumulative returns for different strategies using Plotly.
    
    Args:
        returns_data: Dictionary mapping strategy names to return DataFrames
        benchmark_returns: Optional benchmark returns series
        title: Plot title
        filename: Optional filename to save the plot
    """
    fig = go.Figure()
    
    # Plot each strategy
    for strategy_name, returns_df in returns_data.items():
        if 'Combined' in returns_df.columns:
            # Calculate cumulative returns
            cumulative_returns = (1 + returns_df['Combined']).cumprod() - 1
            
            # Add trace for this strategy
            fig.add_trace(go.Scatter(
                x=cumulative_returns.index,
                y=cumulative_returns,
                mode='lines',
                name=strategy_name
            ))
    
    # Plot benchmark if provided
    if benchmark_returns is not None:
        # Align benchmark with strategy period
        first_date = min([df.index[0] for df in returns_data.values() if not df.empty])
        last_date = max([df.index[-1] for df in returns_data.values() if not df.empty])
        
        benchmark_aligned = benchmark_returns.loc[
            (benchmark_returns.index >= first_date) &
            (benchmark_returns.index <= last_date)
        ]
        
        # Calculate cumulative benchmark returns
        cumulative_benchmark = (1 + benchmark_aligned).cumprod() - 1
        
        # Add trace for benchmark
        fig.add_trace(go.Scatter(
            x=cumulative_benchmark.index,
            y=cumulative_benchmark,
            mode='lines',
            name='Benchmark (SPY)',
            line=dict(color='black', dash='dash')
        ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Cumulative Return',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        yaxis=dict(
            tickformat='.0%',
            hoverformat='.2%'
        ),
        hovermode="x unified",
        template="plotly_white"
    )
    
    # Save figure if filename is provided
    if VISUALIZATION_SETTINGS['save_plots'] and filename:
        fig.write_image(os.path.join(OUTPUT_SETTINGS['plots_directory'], filename))
        # Also save as HTML for interactive viewing
        html_filename = os.path.splitext(filename)[0] + '.html'
        fig.write_html(os.path.join(OUTPUT_SETTINGS['plots_directory'], html_filename))
    
    return fig

def plot_performance_metrics(
    metrics_data: Dict[str, pd.DataFrame],
    metric_name: str,
    title: Optional[str] = None,
    filename: Optional[str] = None
):
    """
    Plots performance metrics for different strategies using Plotly.
    
    Args:
        metrics_data: Dictionary mapping strategy names to metrics DataFrames
        metric_name: Name of the metric to plot
        title: Optional plot title
        filename: Optional filename to save the plot
    """
    # Extract the metric values for each strategy
    strategies = []
    values = []
    
    for strategy_name, metrics_df in metrics_data.items():
        if metric_name in metrics_df.columns and 'Combined' in metrics_df.index:
            strategies.append(strategy_name)
            values.append(metrics_df.loc['Combined', metric_name])
    
    # Create figure
    fig = go.Figure()
    
    # Add bar chart
    fig.add_trace(go.Bar(
        x=strategies,
        y=values,
        text=[f"{v:.2f}" for v in values],
        textposition='auto',
        marker_color=px.colors.qualitative.Plotly[:len(strategies)]
    ))
    
    # Update layout
    fig.update_layout(
        title=title if title else f'{metric_name} by Strategy',
        xaxis_title='Strategy',
        yaxis_title=metric_name,
        template="plotly_white"
    )
    
    # Format y-axis for percentage metrics
    if 'Return' in metric_name or 'rate' in metric_name.lower():
        fig.update_layout(
            yaxis=dict(
                tickformat='.1f%' if 'bps' not in metric_name.lower() else '',
                hoverformat='.2f%' if 'bps' not in metric_name.lower() else ''
            )
        )
    
    # Save figure if filename is provided
    if VISUALIZATION_SETTINGS['save_plots'] and filename:
        fig.write_image(os.path.join(OUTPUT_SETTINGS['plots_directory'], filename))
        # Also save as HTML for interactive viewing
        html_filename = os.path.splitext(filename)[0] + '.html'
        fig.write_html(os.path.join(OUTPUT_SETTINGS['plots_directory'], html_filename))
    
    return fig

def plot_cluster_analysis(
    cluster_history: List[Dict],
    title: str = 'Number of Clusters Over Time',
    filename: Optional[str] = None
):
    """
    Plots the number of clusters over time using Plotly.
    
    Args:
        cluster_history: List of dictionaries with cluster information
        title: Plot title
        filename: Optional filename to save the plot
    """
    dates = [entry['date'] for entry in cluster_history]
    n_clusters = [entry['n_clusters'] for entry in cluster_history]
    
    # Create figure
    fig = go.Figure()
    
    # Add line plot with markers
    fig.add_trace(go.Scatter(
        x=dates,
        y=n_clusters,
        mode='lines+markers',
        name='Number of Clusters',
        marker=dict(size=8)
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Number of Clusters',
        template="plotly_white",
        yaxis=dict(
            # Set y-axis to start from 0
            range=[0, max(n_clusters) * 1.1]
        )
    )
    
    # Save figure if filename is provided
    if VISUALIZATION_SETTINGS['save_plots'] and filename:
        fig.write_image(os.path.join(OUTPUT_SETTINGS['plots_directory'], filename))
        # Also save as HTML for interactive viewing
        html_filename = os.path.splitext(filename)[0] + '.html'
        fig.write_html(os.path.join(OUTPUT_SETTINGS['plots_directory'], html_filename))
    
    return fig

def plot_rolling_metrics(
    returns_data: Dict[str, pd.DataFrame],
    window: int = 252,
    metric: str = 'sharpe',
    title: Optional[str] = None,
    filename: Optional[str] = None
):
    """
    Plots rolling performance metrics for different strategies using Plotly.
    
    Args:
        returns_data: Dictionary mapping strategy names to return DataFrames
        window: Rolling window size in days
        metric: Metric to calculate ('sharpe', 'sortino', 'return')
        title: Optional plot title
        filename: Optional filename to save the plot
    """
    # Create figure
    fig = go.Figure()
    
    # Plot each strategy
    for strategy_name, returns_df in returns_data.items():
        if 'Combined' in returns_df.columns:
            returns_series = returns_df['Combined']
            
            if metric == 'sharpe':
                # Calculate rolling Sharpe ratio
                rolling_return = returns_series.rolling(window=window).mean()
                rolling_std = returns_series.rolling(window=window).std()
                rolling_metric = rolling_return / rolling_std * np.sqrt(252)
                metric_name = 'Sharpe Ratio'
            
            elif metric == 'sortino':
                # Calculate rolling Sortino ratio
                rolling_return = returns_series.rolling(window=window).mean()
                downside_returns = returns_series.copy()
                downside_returns[downside_returns > 0] = 0
                rolling_downside = downside_returns.rolling(window=window).std()
                rolling_metric = rolling_return / rolling_downside * np.sqrt(252)
                rolling_metric.replace([np.inf, -np.inf], np.nan, inplace=True)
                metric_name = 'Sortino Ratio'
            
            elif metric == 'return':
                # Calculate rolling annualized return
                rolling_metric = returns_series.rolling(window=window).mean() * 252
                metric_name = 'Annualized Return'
            
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            # Add trace for this strategy
            fig.add_trace(go.Scatter(
                x=rolling_metric.index,
                y=rolling_metric,
                mode='lines',
                name=strategy_name
            ))
    
    # Update layout
    fig.update_layout(
        title=title if title else f'Rolling {window}-Day {metric_name}',
        xaxis_title='Date',
        yaxis_title=metric_name,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_white",
        hovermode="x unified"
    )
    
    # Format y-axis for percentage metrics
    if metric == 'return':
        fig.update_layout(
            yaxis=dict(
                tickformat='.0%',
                hoverformat='.2%'
            )
        )
    
    # Save figure if filename is provided
    if VISUALIZATION_SETTINGS['save_plots'] and filename:
        fig.write_image(os.path.join(OUTPUT_SETTINGS['plots_directory'], filename))
        # Also save as HTML for interactive viewing
        html_filename = os.path.splitext(filename)[0] + '.html'
        fig.write_html(os.path.join(OUTPUT_SETTINGS['plots_directory'], html_filename))
    
    return fig

def plot_drawdowns(
    returns_data: Dict[str, pd.DataFrame],
    title: str = 'Drawdowns',
    filename: Optional[str] = None
):
    """
    Plots drawdowns for different strategies using Plotly.
    
    Args:
        returns_data: Dictionary mapping strategy names to return DataFrames
        title: Plot title
        filename: Optional filename to save the plot
    """
    # Create figure
    fig = go.Figure()
    
    # Plot each strategy
    for strategy_name, returns_df in returns_data.items():
        if 'Combined' in returns_df.columns:
            returns_series = returns_df['Combined']
            
            # Calculate drawdowns
            cumulative_returns = (1 + returns_series).cumprod()
            running_max = cumulative_returns.cummax()
            drawdowns = (cumulative_returns / running_max) - 1
            
            # Add trace for this strategy
            fig.add_trace(go.Scatter(
                x=drawdowns.index,
                y=drawdowns,
                mode='lines',
                name=strategy_name
            ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Drawdown',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_white",
        hovermode="x unified",
        yaxis=dict(
            tickformat='.0%',
            hoverformat='.2%'
        )
    )
    
    # Save figure if filename is provided
    if VISUALIZATION_SETTINGS['save_plots'] and filename:
        fig.write_image(os.path.join(OUTPUT_SETTINGS['plots_directory'], filename))
        # Also save as HTML for interactive viewing
        html_filename = os.path.splitext(filename)[0] + '.html'
        fig.write_html(os.path.join(OUTPUT_SETTINGS['plots_directory'], html_filename))
    
    return fig

def plot_monthly_returns_heatmap(
    returns_data: Dict[str, pd.DataFrame],
    strategy_name: str,
    title: Optional[str] = None,
    filename: Optional[str] = None
):
    """
    Plots a heatmap of monthly returns for a specific strategy using Plotly.
    
    Args:
        returns_data: Dictionary mapping strategy names to return DataFrames
        strategy_name: Name of the strategy to plot
        title: Optional plot title
        filename: Optional filename to save the plot
    """
    if strategy_name not in returns_data or 'Combined' not in returns_data[strategy_name].columns:
        print(f"Strategy {strategy_name} not found in returns data")
        return None
    
    returns_series = returns_data[strategy_name]['Combined']
    
    # Convert to monthly returns
    monthly_returns = returns_series.resample('M').apply(lambda x: (1 + x).prod() - 1)
    
    # Create a pivot table of monthly returns by year and month
    monthly_pivot = pd.DataFrame({
        'Year': monthly_returns.index.year,
        'Month': monthly_returns.index.month,
        'Return': monthly_returns.values
    })
    
    monthly_pivot = monthly_pivot.pivot(index='Year', columns='Month', values='Return')
    
    # Replace month numbers with names
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_pivot.columns = [month_names[i-1] for i in monthly_pivot.columns]
    
    # Create figure
    fig = go.Figure(data=go.Heatmap(
        z=monthly_pivot.values,
        x=monthly_pivot.columns,
        y=monthly_pivot.index,
        colorscale='RdYlGn',  # Red for negative, green for positive
        zmid=0,  # Center the color scale at 0
        text=[[f"{val:.2%}" for val in row] for row in monthly_pivot.values],
        texttemplate="%{text}",
        hovertemplate='Year: %{y}<br>Month: %{x}<br>Return: %{z:.2%}<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title=title if title else f'Monthly Returns: {strategy_name}',
        xaxis_title='Month',
        yaxis_title='Year',
        template="plotly_white",
        height=600
    )
    
    # Save figure if filename is provided
    if VISUALIZATION_SETTINGS['save_plots'] and filename:
        fig.write_image(os.path.join(OUTPUT_SETTINGS['plots_directory'], filename))
        # Also save as HTML for interactive viewing
        html_filename = os.path.splitext(filename)[0] + '.html'
        fig.write_html(os.path.join(OUTPUT_SETTINGS['plots_directory'], html_filename))
    
    return fig

def plot_correlation_matrix(
    correlation_matrix: pd.DataFrame,
    title: str = 'Correlation Matrix',
    filename: Optional[str] = None
):
    """
    Plots a heatmap of the correlation matrix using Plotly.
    
    Args:
        correlation_matrix: DataFrame with correlation matrix
        title: Plot title
        filename: Optional filename to save the plot
    """
    # Create figure
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.index,
        colorscale='RdBu_r',  # Red for negative, blue for positive
        zmid=0,  # Center the color scale at 0
        zmin=-1,
        zmax=1,
        text=[[f"{val:.2f}" for val in row] for row in correlation_matrix.values],
        hovertemplate='%{y} & %{x}<br>Correlation: %{z:.2f}<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=800,
        width=800,
        xaxis=dict(
            tickangle=-45,
            tickfont=dict(size=10)
        ),
        yaxis=dict(
            tickfont=dict(size=10)
        )
    )
    
    # Save figure if filename is provided
    if VISUALIZATION_SETTINGS['save_plots'] and filename:
        fig.write_image(os.path.join(OUTPUT_SETTINGS['plots_directory'], filename))
        # Also save as HTML for interactive viewing
        html_filename = os.path.splitext(filename)[0] + '.html'
        fig.write_html(os.path.join(OUTPUT_SETTINGS['plots_directory'], html_filename))
    
    return fig

def plot_clusters(
    correlation_matrix: pd.DataFrame,
    labels: np.ndarray,
    ticker_names: Optional[List[str]] = None,
    title: str = 'Correlation Matrix with Clusters',
    filename: Optional[str] = None
):
    """
    Plots the correlation matrix with stocks sorted by cluster using Plotly.
    
    Args:
        correlation_matrix: DataFrame with correlation matrix
        labels: Array of cluster assignments
        ticker_names: Optional list of ticker names (if not in correlation_matrix)
        title: Plot title
        filename: Optional filename to save the plot
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
    
    # Create figure
    fig = go.Figure(data=go.Heatmap(
        z=sorted_corr.values,
        x=sorted_corr.columns,
        y=sorted_corr.index,
        colorscale='RdBu_r',  # Red for negative, blue for positive
        zmid=0,  # Center the color scale at 0
        zmin=-1,
        zmax=1,
        text=[[f"{val:.2f}" for val in row] for row in sorted_corr.values],
        hovertemplate='%{y} & %{x}<br>Correlation: %{z:.2f}<extra></extra>'
    ))
    
    # Add cluster boundaries
    cluster_boundaries = []
    current_cluster = labels[sorted_indices[0]]
    
    for i, idx in enumerate(sorted_indices[1:], 1):
        if labels[idx] != current_cluster:
            # Add a vertical line
            fig.add_shape(
                type="line",
                x0=i - 0.5,
                y0=0 - 0.5,
                x1=i - 0.5,
                y1=len(sorted_tickers) - 0.5,
                line=dict(color="black", width=2)
            )
            
            # Add a horizontal line
            fig.add_shape(
                type="line",
                x0=0 - 0.5,
                y0=i - 0.5,
                x1=len(sorted_tickers) - 0.5,
                y1=i - 0.5,
                line=dict(color="black", width=2)
            )
            
            current_cluster = labels[idx]
    
    # Update layout
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=800,
        width=800,
        xaxis=dict(
            tickangle=-45,
            tickfont=dict(size=10)
        ),
        yaxis=dict(
            tickfont=dict(size=10)
        )
    )
    
    # Save figure if filename is provided
    if VISUALIZATION_SETTINGS['save_plots'] and filename:
        fig.write_image(os.path.join(OUTPUT_SETTINGS['plots_directory'], filename))
        # Also save as HTML for interactive viewing
        html_filename = os.path.splitext(filename)[0] + '.html'
        fig.write_html(os.path.join(OUTPUT_SETTINGS['plots_directory'], html_filename))
    
    return fig

def plot_transaction_costs(
    portfolio_returns: pd.DataFrame,
    title: str = 'Transaction Costs',
    filename: Optional[str] = None
):
    """
    Plots transaction costs over time using Plotly.
    
    Args:
        portfolio_returns: DataFrame with portfolio returns including transaction costs
        title: Plot title
        filename: Optional filename to save the plot
    """
    # Extract transaction cost columns
    tc_columns = [col for col in portfolio_returns.columns if col.endswith('_TC')]
    
    if not tc_columns:
        print("No transaction cost columns found in portfolio returns")
        return None
    
    # Create figure
    fig = go.Figure()
    
    # Add transaction costs for each component
    for col in tc_columns:
        # Skip if all values are zero
        if portfolio_returns[col].sum() == 0:
            continue
            
        fig.add_trace(go.Bar(
            x=portfolio_returns.index,
            y=portfolio_returns[col],
            name=col.replace('_TC', '')
        ))
    
    # Add total transaction costs
    total_tc = portfolio_returns[tc_columns].sum(axis=1)
    
    fig.add_trace(go.Scatter(
        x=total_tc.index,
        y=total_tc.cumsum(),
        mode='lines',
        name='Cumulative Total TC',
        yaxis='y2',
        line=dict(color='black', width=2)
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis=dict(
            title='Daily Transaction Cost',
            tickformat='.3%',
            hoverformat='.4%',
            side='left'
        ),
        yaxis2=dict(
            title='Cumulative Transaction Cost',
            tickformat='.2%',
            hoverformat='.3%',
            overlaying='y',
            side='right'
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_white",
        hovermode="x unified"
    )
    
    # Save figure if filename is provided
    if VISUALIZATION_SETTINGS['save_plots'] and filename:
        fig.write_image(os.path.join(OUTPUT_SETTINGS['plots_directory'], filename))
        # Also save as HTML for interactive viewing
        html_filename = os.path.splitext(filename)[0] + '.html'
        fig.write_html(os.path.join(OUTPUT_SETTINGS['plots_directory'], html_filename))
    
    return fig

def create_performance_report(
    returns_data: Dict[str, pd.DataFrame],
    metrics_data: Dict[str, pd.DataFrame],
    benchmark_returns: Optional[pd.Series] = None,
    output_file: Optional[str] = None
):
    """
    Creates a comprehensive performance report.
    
    Args:
        returns_data: Dictionary mapping strategy names to return DataFrames
        metrics_data: Dictionary mapping strategy names to metrics DataFrames
        benchmark_returns: Optional benchmark returns series
        output_file: Optional file path to save the report
    """
    # Create output directory if it doesn't exist
    if output_file and not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    
    # Prepare report data
    report = []
    
    # Report header
    report.append("# Statistical Arbitrage Performance Report")
    report.append(f"## Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Performance summary table
    report.append("## Performance Summary")
    report.append("")
    
    summary_table = []
    headers = ["Strategy", "Ann. Return (%)", "Sharpe Ratio", "Sortino Ratio", "Max Drawdown (%)", "Win Rate (%)", "Ann. TC (%)"]
    summary_table.append("| " + " | ".join(headers) + " |")
    summary_table.append("| " + " | ".join(["---" for _ in headers]) + " |")
    
    for strategy_name, metrics_df in metrics_data.items():
        if 'Combined' in metrics_df.index:
            # Transaction costs
            tc_pct = metrics_df.loc['Combined', 'Annualized Transaction Costs (%)'] if 'Annualized Transaction Costs (%)' in metrics_df.columns else 0.0
            
            row = [
                strategy_name,
                f"{metrics_df.loc['Combined', 'Annualized Return (%)']:.2f}",
                f"{metrics_df.loc['Combined', 'Sharpe Ratio']:.2f}",
                f"{metrics_df.loc['Combined', 'Sortino Ratio']:.2f}",
                f"{metrics_df.loc['Combined', 'Maximum Drawdown (%)']:.2f}",
                f"{metrics_df.loc['Combined', 'Win Rate (%)']:.2f}",
                f"{tc_pct:.2f}"
            ]
            summary_table.append("| " + " | ".join(row) + " |")
    
    # Add benchmark if available
    if benchmark_returns is not None:
        # Calculate benchmark metrics
        ann_return = (1 + benchmark_returns.mean()) ** 252 - 1
        sharpe = benchmark_returns.mean() / benchmark_returns.std() * np.sqrt(252)
        downside_returns = benchmark_returns[benchmark_returns < 0]
        sortino = benchmark_returns.mean() / downside_returns.std() * np.sqrt(252)
        cumulative_returns = (1 + benchmark_returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdowns = (cumulative_returns / running_max) - 1
        max_drawdown = drawdowns.min()
        win_rate = (benchmark_returns > 0).mean()
        
        row = [
            "Benchmark (SPY)",
            f"{ann_return * 100:.2f}",
            f"{sharpe:.2f}",
            f"{sortino:.2f}",
            f"{max_drawdown * 100:.2f}",
            f"{win_rate * 100:.2f}",
            "0.00"  # No transaction costs for benchmark
        ]
        summary_table.append("| " + " | ".join(row) + " |")
    
    report.extend(summary_table)
    report.append("")
    
    # Transaction costs analysis
    report.append("## Transaction Costs Analysis")
    report.append("")
    
    tc_table = []
    tc_headers = ["Strategy", "Total TC (%)", "Ann. TC (%)", "Avg TC per Trade (bps)", "Ann. Turnover (x)"]
    tc_table.append("| " + " | ".join(tc_headers) + " |")
    tc_table.append("| " + " | ".join(["---" for _ in tc_headers]) + " |")
    
    for strategy_name, metrics_df in metrics_data.items():
        if 'Combined' in metrics_df.index and 'Total Transaction Costs (%)' in metrics_df.columns:
            row = [
                strategy_name,
                f"{metrics_df.loc['Combined', 'Total Transaction Costs (%)']:.2f}",
                f"{metrics_df.loc['Combined', 'Annualized Transaction Costs (%)']:.2f}",
                f"{metrics_df.loc['Combined', 'Avg Transaction Cost per Trade (bps)']:.2f}",
                f"{metrics_df.loc['Combined', 'Annualized Turnover (x)']:.2f}"
            ]
            tc_table.append("| " + " | ".join(row) + " |")
    
    report.extend(tc_table)
    report.append("")
    
    # Strategy details section
    for strategy_name, returns_df in returns_data.items():
        report.append(f"## {strategy_name} Details")
        report.append("")
        
        if 'Combined' in returns_df.columns:
            returns_series = returns_df['Combined']
            
            # Calculate annual returns
            annual_returns = returns_series.resample('Y').apply(lambda x: (1 + x).prod() - 1)
            
            report.append("### Annual Returns")
            report.append("")
            
            annual_table = []
            annual_table.append("| Year | Return (%) |")
            annual_table.append("| --- | ---: |")
            
            for year, ret in annual_returns.items():
                annual_table.append(f"| {year.year} | {ret * 100:.2f} |")
            
            report.extend(annual_table)
            report.append("")
            
            # Calculate rolling metrics
            rolling_window = 252  # 1 year
            
            # Rolling Sharpe
            rolling_return = returns_series.rolling(window=rolling_window).mean()
            rolling_std = returns_series.rolling(window=rolling_window).std()
            rolling_sharpe = rolling_return / rolling_std * np.sqrt(252)
            
            # Rolling Sortino
            downside_returns = returns_series.copy()
            downside_returns[downside_returns > 0] = 0
            rolling_downside = downside_returns.rolling(window=rolling_window).std()
            rolling_sortino = rolling_return / rolling_downside * np.sqrt(252)
            rolling_sortino.replace([np.inf, -np.inf], np.nan, inplace=True)
            
            report.append("### Rolling Metrics (1-Year Window)")
            report.append("")
            
            rolling_table = []
            rolling_table.append("| Metric | Current | Min | Max | Average |")
            rolling_table.append("| --- | ---: | ---: | ---: | ---: |")
            
            rolling_table.append(f"| Sharpe Ratio | {rolling_sharpe.iloc[-1]:.2f} | {rolling_sharpe.min():.2f} | {rolling_sharpe.max():.2f} | {rolling_sharpe.mean():.2f} |")
            rolling_table.append(f"| Sortino Ratio | {rolling_sortino.iloc[-1]:.2f} | {rolling_sortino.min():.2f} | {rolling_sortino.max():.2f} | {rolling_sortino.mean():.2f} |")
            
            report.extend(rolling_table)
            report.append("")
        
        # Add transaction costs details
        tc_col = f"Combined_TC"
        if tc_col in returns_df.columns:
            tc_series = returns_df[tc_col]
            
            report.append("### Transaction Costs")
            report.append("")
            
            tc_stats_table = []
            tc_stats_table.append("| Metric | Value |")
            tc_stats_table.append("| --- | ---: |")
            
            tc_stats_table.append(f"| Total TC (%) | {tc_series.sum() * 100:.2f} |")
            tc_stats_table.append(f"| Average Daily TC (bps) | {tc_series.mean() * 10000:.2f} |")
            tc_stats_table.append(f"| Max Daily TC (bps) | {tc_series.max() * 10000:.2f} |")
            tc_stats_table.append(f"| Number of Rebalances | {(tc_series > 0).sum()} |")
            
            report.extend(tc_stats_table)
            report.append("")
    
    # Combine report content
    report_content = "\n".join(report)
    
    # Save report if output file is provided
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report_content)
    
    return report_content

if __name__ == "__main__":
    # Example usage
    setup_visualization()
    
    # Mock data for testing
    dates = pd.date_range(start='2000-01-01', end='2022-12-31', freq='B')
    
    # Create mock returns data
    np.random.seed(42)
    mock_returns1 = pd.DataFrame({
        'Combined': np.random.normal(0.0005, 0.01, len(dates)),
        'Combined_TC': np.random.normal(0.00005, 0.0001, len(dates)).clip(0, None)
    }, index=dates)
    
    mock_returns2 = pd.DataFrame({
        'Combined': np.random.normal(0.0003, 0.008, len(dates)),
        'Combined_TC': np.random.normal(0.00004, 0.0001, len(dates)).clip(0, None)
    }, index=dates)
    
    # Create mock benchmark returns
    mock_benchmark = pd.Series(
        np.random.normal(0.0004, 0.012, len(dates)),
        index=dates
    )
    
    # Create mock metrics data
    mock_metrics1 = pd.DataFrame({
        'Annualized Return (%)': [12.5],
        'Sharpe Ratio': [1.2],
        'Sortino Ratio': [1.8],
        'Maximum Drawdown (%)': [-15.3],
        'Win Rate (%)': [55.2],
        'Total Transaction Costs (%)': [2.5],
        'Annualized Transaction Costs (%)': [1.1],
        'Avg Transaction Cost per Trade (bps)': [5.0],
        'Annualized Turnover (x)': [2.2]
    }, index=['Combined'])
    
    mock_metrics2 = pd.DataFrame({
        'Annualized Return (%)': [9.8],
        'Sharpe Ratio': [0.9],
        'Sortino Ratio': [1.5],
        'Maximum Drawdown (%)': [-12.1],
        'Win Rate (%)': [53.7],
        'Total Transaction Costs (%)': [2.0],
        'Annualized Transaction Costs (%)': [0.9],
        'Avg Transaction Cost per Trade (bps)': [5.0],
        'Annualized Turnover (x)': [1.8]
    }, index=['Combined'])
    
    # Create mock cluster history
    mock_cluster_history = []
    for i, date in enumerate(dates[::20]):  # Every 20 days
        mock_cluster_history.append({
            'date': date,
            'n_clusters': 10 + int(np.sin(i / 10) * 5)
        })
    
    # Package mock data
    returns_data = {
        'SPONGE Clustering': mock_returns1,
        'Industry Benchmark': mock_returns2
    }
    
    metrics_data = {
        'SPONGE Clustering': mock_metrics1,
        'Industry Benchmark': mock_metrics2
    }
    
    # Test visualizations
    plot_cumulative_returns(
        returns_data, 
        mock_benchmark,
        title='Strategy Cumulative Returns',
        filename='cumulative_returns.png'
    )
    
    plot_performance_metrics(
        metrics_data,
        'Sharpe Ratio',
        title='Sharpe Ratio Comparison',
        filename='sharpe_ratio_comparison.png'
    )
    
    plot_cluster_analysis(
        mock_cluster_history,
        title='Number of Clusters Over Time',
        filename='cluster_analysis.png'
    )
    
    plot_rolling_metrics(
        returns_data,
        window=252,
        metric='sharpe',
        title='Rolling 1-Year Sharpe Ratio',
        filename='rolling_sharpe.png'
    )
    
    plot_drawdowns(
        returns_data,
        title='Strategy Drawdowns',
        filename='drawdowns.png'
    )
    
    plot_monthly_returns_heatmap(
        returns_data,
        'SPONGE Clustering',
        title='Monthly Returns: SPONGE Clustering',
        filename='monthly_returns_heatmap.png'
    )
    
    plot_transaction_costs(
        mock_returns1,
        title='Transaction Costs: SPONGE Clustering',
        filename='transaction_costs.png'
    )
    
    # Create performance report
    report = create_performance_report(
        returns_data,
        metrics_data,
        mock_benchmark,
        output_file='results/performance_report.md'
    )
    
    print("Visualizations created successfully!")
