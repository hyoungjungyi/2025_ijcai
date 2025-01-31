import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
import os
import matplotlib.dates as mdates
import seaborn as sns
def calculate_portfolio_metrics(portfolio_values, risk_free_rate=0.01, freq="daily", pred_len=None, total_periods=None,dynamic_annual_factor=None):
    """
    Calculate common portfolio metrics for performance evaluation.
    Supports custom pred_len and total_periods for non-standard frequencies.
    """
    # Determine annualization factor
    if dynamic_annual_factor is not None:
        annual_factor = dynamic_annual_factor
    elif pred_len is not None and total_periods is not None:
        steps_per_year = 252.0 *(len(portfolio_values) - 1 ) / total_periods
        annual_factor = steps_per_year
    else:
        if freq == "daily":
            annual_factor = 252
        elif freq == "weekly":
            annual_factor = 52
        elif freq == "monthly":
            annual_factor = 12
        else:
            raise ValueError(
                "Frequency must be 'daily', 'weekly', 'monthly', or pred_len/total_periods must be provided.")

    # Periodic returns
    periodic_returns = np.diff(portfolio_values) / portfolio_values[:-1]

    # Total Return
    total_return = (portfolio_values[-1] / portfolio_values[0]) - 1

    # Annualized Return (CAGR)
    years = len(portfolio_values) / annual_factor
    cagr = (portfolio_values[-1] / portfolio_values[0]) ** (1 / years) - 1

    # Volatility (Standard Deviation of Returns)
    volatility = np.std(periodic_returns) * np.sqrt(annual_factor)

    # Sharpe Ratio
    sharpe_ratio = (np.mean(periodic_returns) - risk_free_rate / annual_factor) / np.std(periodic_returns)

    # Sortino Ratio
    downside_returns = periodic_returns[periodic_returns < 0]
    downside_std = np.std(downside_returns)
    sortino_ratio = (np.mean(periodic_returns) - risk_free_rate / annual_factor) / downside_std

    # Max Drawdown
    drawdown = portfolio_values / np.maximum.accumulate(portfolio_values) - 1
    max_drawdown = drawdown.min()

    # Calmar Ratio
    calmar_ratio = cagr / abs(max_drawdown)

    return {
        "Total Return (%)": total_return * 100,
        "CAGR (%)": cagr * 100,
        "Sharpe Ratio": sharpe_ratio,
        "Sortino Ratio": sortino_ratio,
        "Calmar Ratio": calmar_ratio,
        "Max Drawdown (%)": max_drawdown * 100,
        "Volatility (%)": volatility * 100,
    }


def fetch_index_data(index, start_date, end_date):
    """Fetch index data for a given date range."""
    print(f"Fetching index data from {start_date} to {end_date}...")
    end_date += pd.Timedelta(days=1)  # Include end date
    index = yf.download(f'{index}', start=pd.to_datetime(start_date), end=pd.to_datetime(end_date))
    index = index[['Close']].reset_index()
    index['Date'] = pd.to_datetime(index['Date'])
    index.rename(columns={'Close': 'close', 'Date': 'date'}, inplace=True)
    return index


def run_backtest(data, index_data, start_date, end_date, fee_rate=0.00, external_portfolio=None,external_dates=None,pred_len=None,total_periods=None,folder_path=None,dynamic_annual_factor=None ):
    """Run backtest for the given data and date range."""

    data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]
    data['date'] = pd.to_datetime(data['date'])
    def analyze_rebalancing_changes(data, rebalancing_dates):
        """
        Analyze how portfolio composition changes at rebalancing points.
        Args:
            data (DataFrame): Data with prices and returns for each stock.
            rebalancing_dates (list): Dates when rebalancing occurred.
        Returns:
            DataFrame: Portfolio composition and returns at each rebalancing point.
        """
        results = []
        for i in range(1, len(rebalancing_dates)):
            current_date = rebalancing_dates[i]
            previous_date = rebalancing_dates[i - 1]

            # Filter data for the current and previous rebalancing dates
            current_data = data[data['date'] == current_date].set_index('tic')['close']
            previous_data = data[data['date'] == previous_date].set_index('tic')['close']

            # Calculate returns and portfolio weights
            returns = current_data / previous_data - 1
            portfolio_weights = 1 / len(current_data)

            results.append({
                'Date': current_date,
                'Average Return': returns.mean(),
                'Max Return': returns.max(),
                'Min Return': returns.min(),
                'Volatility': returns.std()
            })

        return pd.DataFrame(results)




    # Helper functions
    def get_rebalancing_dates_fixed(data, freq):
        """Get rebalancing dates based on frequency (weekly or monthly)."""
        if freq == 'weekly':
            return data['date'].groupby(data['date'].dt.to_period('W')).min().tolist()
        elif freq == 'monthly':
            return data['date'].groupby(data['date'].dt.to_period('M')).min().tolist()
        else:
            raise ValueError("Frequency must be 'weekly' or 'monthly'.")

    def rebalancing_strategy_fixed(data, rebalancing_dates, fee_rate):
        """Calculate portfolio value over time with proper initialization for rebalancing."""
        current_portfolio_value = initial_investment
        portfolio_values = [current_portfolio_value]  # To track history

        for i in range(1, len(rebalancing_dates)):
            current_date = rebalancing_dates[i]
            previous_date = rebalancing_dates[i - 1]
            current_data = data[data['date'] == current_date].set_index('tic')
            previous_data = data[data['date'] == previous_date].set_index('tic')

            if current_data.empty or previous_data.empty:
                continue

            # Calculate portfolio return based on equal weights
            weights = 1 / len(previous_data)
            returns = current_data['close'] / previous_data['close'] - 1
            portfolio_return = (returns * weights).sum()

            # Apply fees based on the traded value
            traded_value = current_portfolio_value
            buy_fee = traded_value * fee_rate
            sell_fee = (current_portfolio_value * (1 + portfolio_return)) * fee_rate

            # Update portfolio value after accounting for returns and fees
            current_portfolio_value = current_portfolio_value * (1 + portfolio_return) - buy_fee - sell_fee
            portfolio_values.append(current_portfolio_value)

        return portfolio_values, rebalancing_dates[:len(portfolio_values)]

    # Define initial investment
    initial_investment = 1.0

    # Extract rebalancing dates
    weekly_dates = get_rebalancing_dates_fixed(data, 'weekly')
    monthly_dates = get_rebalancing_dates_fixed(data, 'monthly')

    # index data values
    index_portfolio_values = []  # Start empty
    # Calculate daily returns for DJ30 index
    index_data = index_data.copy()
    index_data.loc[:,'daily_return'] = index_data['close'].pct_change()

    for daily_return in index_data['daily_return']:  # Skip the first NaN value
        # Calculate portfolio value
        if not index_portfolio_values:  # If the list is empty, start with initial investment
            index_portfolio_values.append(initial_investment)
            continue
        index_portfolio_values.append(index_portfolio_values[-1] * (1 + daily_return))

    index_dates = index_data['date']  # Skip the first day for alignment




    ## Buy-and-Hold Strategy
    initial_prices = data[data['date'] == data['date'].min()].set_index('tic')['close']

    # Create a list to store daily portfolio values
    bh_portfolio_values = []

    # Iterate through each unique date to calculate daily portfolio value
    for date in data['date'].unique():
        daily_prices = data[data['date'] == date].set_index('tic')['close']

        # Calculate price change ratio for each asset
        price_change_ratio = daily_prices / initial_prices

        # Calculate daily portfolio value as average price change ratio
        daily_portfolio_value = initial_investment * price_change_ratio.mean()
        bh_portfolio_values.append(daily_portfolio_value)

    ## Daily portfolio value calculation
    daily_portfolio_values = []  # Start empty
    daily_portfolio_value = initial_investment  # Initialize daily portfolio value
    data = data.copy()
    data['daily_return'] = data.groupby('tic')['close'].pct_change()  # Daily returns per stock
    for date, group in data.groupby('date'):
        if group['daily_return'].isna().any():
            daily_portfolio_values.append(initial_investment)
            continue
        daily_return = group['daily_return'].mean()

        # Reinvest gains/losses
        traded_value = daily_portfolio_value  # Entire portfolio value is traded (before return)
        portfolio_return = daily_portfolio_value * daily_return  # Value change due to return

        # Apply fees
        buy_fee = traded_value * fee_rate  # Buy fee
        sell_fee = (daily_portfolio_value + portfolio_return) * fee_rate  # Sell fee

        # Update portfolio value after returns and fees
        daily_portfolio_value = daily_portfolio_value + portfolio_return - buy_fee - sell_fee

        # Append updated portfolio value
        daily_portfolio_values.append(daily_portfolio_value)

    if external_portfolio is not None:
        external_metrics = calculate_portfolio_metrics(external_portfolio,  pred_len=pred_len, total_periods=total_periods,dynamic_annual_factor=dynamic_annual_factor)
    else:
        external_metrics = None


    ## Rebalancing strategies
    weekly_values, weekly_dates_used = rebalancing_strategy_fixed(data, weekly_dates, fee_rate)
    monthly_values, monthly_dates_used = rebalancing_strategy_fixed(data, monthly_dates, fee_rate)
    # rebalancing_analysis = analyze_rebalancing_changes(data, monthly_dates_used)
    # print(rebalancing_analysis)
    # Calculate metrics for each portfolio strategy
    index_metrics = calculate_portfolio_metrics(index_portfolio_values,freq='daily')
    # Calculate metrics for Buy-and-Hold Strategy
    bh_metrics = calculate_portfolio_metrics(bh_portfolio_values,freq='daily')
    # Daily Rebalancing
    daily_metrics = calculate_portfolio_metrics(daily_portfolio_values, freq="daily")
    # Weekly Rebalancing
    weekly_metrics = calculate_portfolio_metrics(weekly_values, freq="weekly")
    # Monthly Rebalancing
    monthly_metrics = calculate_portfolio_metrics(monthly_values, freq="monthly")

    # Combine results into a DataFrame
    metrics_summary = {
        "Metric": [
            "Total Return (%)", "CAGR (%)",
            "Sharpe Ratio", "Sortino Ratio","Calmar Ratio","Max Drawdown (%)", "Volatility (%)"],
        "Index": list(index_metrics.values()),
        "Buy-and-Holds": list(bh_metrics.values()),
        "Daily Rebalancing": list(daily_metrics.values()),
        "Weekly Rebalancing": list(weekly_metrics.values()),
        "Monthly Rebalancing": list(monthly_metrics.values())
    }
    if external_portfolio is not None:
        metrics_summary["External Portfolio"] = list(external_metrics.values())
    metrics_summary_df = pd.DataFrame(metrics_summary)

    # Transpose for better readability
    metrics_summary_df_transposed = metrics_summary_df.set_index("Metric").transpose()
    pd.set_option('display.max_columns', None)
    print(metrics_summary_df_transposed)

    # Plot portfolio values
    sns.set(style="whitegrid")
    plt.figure(figsize=(14, 8))

    plt.plot(index_dates, index_portfolio_values, label="Index", color="orange", linestyle="--", linewidth=2)
    plt.plot(data['date'].unique(), bh_portfolio_values, label="Buy-and-Hold", color="purple", linewidth=2)
    plt.plot(data['date'].unique(), daily_portfolio_values, label="Daily Rebalancing", color="blue", linewidth=2,
             linestyle="-")
    if external_portfolio is not None:
        plt.plot(external_dates, external_portfolio, label="External Portfolio", color="cyan", linewidth=2,
                 linestyle="-.")
    plt.plot(weekly_dates_used, weekly_values, label="Weekly Rebalancing", color="green", linewidth=2)
    plt.plot(monthly_dates_used, monthly_values, label="Monthly Rebalancing", color="red", linewidth=2, linestyle=":")

    plt.title("Portfolio Value Comparison", fontsize=16, color="darkblue")
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Portfolio Value", fontsize=14)
    plt.xticks(fontsize=12, rotation=45)
    plt.yticks(fontsize=12)

    # Add gridlines
    plt.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.7)

    # Adjust x-axis date format
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    plt.gca().xaxis.set_major_locator(locator)
    plt.gca().xaxis.set_major_formatter(formatter)

    # Add legend
    plt.legend(loc="upper left", fontsize=12, frameon=True, shadow=True, borderpad=1)

    # Tight layout
    plt.tight_layout()
    if folder_path is not None:
        plt.savefig(f"{folder_path}/portfolio_comparison_plot.png")
        plt.close()
        metrics_summary_df_transposed.to_csv(os.path.join(folder_path, 'metrics_comparison.csv'))
    else:
        plt.show()
    return metrics_summary


# Example usage
if __name__ == '__main__':

    data = pd.read_csv('../data/dj30/data.csv', index_col=0)
    data['date'] = pd.to_datetime(data['date'])
    index_data = fetch_index_data('^DJI', '2012-01-01', '2022-01-01')

    run_backtest(data, index_data, '2020-01-01', '2022-12-31',fee_rate=0.0)

# def analyze_volatility_contribution(data, weights):
#     """
#     Analyze how much each stock contributes to overall portfolio volatility.
#     Args:
#         data (DataFrame): Data with daily returns for each stock.
#         weights (Series): Portfolio weights for each stock.
#     Returns:
#         DataFrame: Contribution to volatility for each stock.
#     """
#     # Calculate daily returns for each stock
#     daily_returns = data.pivot(index='date', columns='tic', values='adjclose').pct_change()
#
#     # Calculate individual stock volatility
#     stock_volatility = daily_returns.std()
#
#     # Calculate contribution to portfolio volatility
#     contribution = (weights * stock_volatility) / (weights * stock_volatility).sum()
#
#     return pd.DataFrame({'Stock': contribution.index, 'Volatility Contribution': contribution.values})
#
# # Example usage
# initial_weights = 1 / len(data['tic'].unique())  # Equal weights
# weights = pd.Series([initial_weights] * len(data['tic'].unique()), index=data['tic'].unique())
# volatility_contribution = analyze_volatility_contribution(data, weights)
# print(volatility_contribution)
#
# # Analyze portfolio changes at rebalancing points
#
# # Analyze correlation between stocks
# def analyze_correlation(data):
#     """
#     Analyze correlation between stocks in the portfolio.
#     Args:
#         data (DataFrame): Data with prices for each stock.
#     Returns:
#         DataFrame: Correlation matrix.
#     """
#     # Pivot data to get stock prices as columns
#     prices = data.pivot(index='date', columns='tic', values='adjclose')
#
#     # Calculate daily returns
#     daily_returns = prices.pct_change()
#
#     # Calculate correlation matrix
#     correlation_matrix = daily_returns.corr()
#
#     return correlation_matrix
#
# # Example usage
# correlation_matrix = analyze_correlation(data)
# print("Correlation Matrix:")
# print(correlation_matrix)
