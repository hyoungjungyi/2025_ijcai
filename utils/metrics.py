import numpy as np
import math

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe


def Cagr(returns,rf=0.):
    """
    Calculates the communicative annualized growth return
    (CAGR%) of access returns

    If rf is non-zero, you must specify periods.
    In this case, rf is assumed to be expressed in yearly (annualized) terms
    """
    total = (returns.add(1).prod()-1)

    years = (returns.index[-1] - returns.index[0]).days / 365.
    res = abs(total +1.0)**(1.0 / years) -1
    return res

def Sharpe(returns,rf=0.,periods=252,annualize=True):
    """
    Calculates the sharpe ratio of access returns

    If rf is non-zero, you must specify periods.
    In this case, rf is assumed to be expressed in yearly (annualized) terms

    Args:
        * returns (Series, DataFrame): Input return series
        * rf (float): Risk-free rate expressed as a yearly (annualized) return
        * periods (int): Freq. of returns (252/365 for daily, 12 for monthly)
        * annualize: return annualize sharpe?
        * smart: return smart sharpe ratio
    """
    if rf !=0 and periods is None:
        raise Exception('Must provide periods if rf !=0')

    divisor = returns.std(ddof=1)
    res = returns.mean() / divisor

    if annualize:
        return res*np.sqrt(1 if periods is None else periods)

    return res

def Sortino(returns,rf=0.,periods=252,annualize=True):
    """
    Calculates the sortino ratio of access returns

    If rf is non-zero, you must specify periods.
    In this case, rf is assumed to be expressed in yearly (annualized) terms

    Calculation is based on this paper by Red Rock Capital
    http://www.redrockcapital.com/Sortino__A__Sharper__Ratio_Red_Rock_Capital.pdf
    """
    if rf != 0 and periods is None:
        raise Exception('Must provide periods if rf != 0')

    downside = np.sqrt((returns[returns <0]**2).sum() / len(returns))
    res = returns.mean() /downside

    if annualize:
        return res * np.sqrt(1 if periods is None else periods)

    return res

def Omega(returns,rf=0.0, required_return=0.0, periods=252):
    """
    Determines the Omega ratio of a strategy.
    See https://en.wikipedia.org/wiki/Omega_ratio for more details.
    """
    if periods ==1:
        return_threshold = required_return
    else:
        return_threshold = (1 + required_return) **(1. / periods) -1

    returns_less_thresh = returns - return_threshold
    numer = returns_less_thresh[returns_less_thresh >0.0].sum()
    denom = -1.0*returns_less_thresh[returns_less_thresh <0.0].sum()

    if denom >0.0:
        return numer / denom
    return np.nan

def port_metric(returns):
    cagr = Cagr(returns)
    sharpe = Sharpe(returns)
    sortino = Sortino(returns)
    omega = Omega(returns)
    return cagr,sharpe,sortino,omega


def calculate_metrics(trade_ror, trade_mode, MAR=0.):
    """
    Based on metric descriptions at AlphaStock
    """
    cm_return = (trade_ror+1).cumprod()-1
    agent_wealth = 1e5 +1e5 * (cm_return)

    if trade_mode == 'D':
        Ny = 251
    elif trade_mode == 'W':
        Ny = 50
    elif trade_mode == 'M':
        Ny = 12
    else:
        assert ValueError, 'Please check the trading mode'

    AT = np.mean(trade_ror)
    VT = np.std(trade_ror)

    APR = AT * Ny
    AVOL = VT * math.sqrt(Ny)
    ASR = APR / AVOL
    drawdown = (np.maximum.accumulate(agent_wealth) - agent_wealth) /\
                     np.maximum.accumulate(agent_wealth)
    MDD = np.max(drawdown)
    CR = APR / MDD

    tmp1 = np.sum(((np.clip(MAR-trade_ror, 0., math.inf))**2)) / \
           np.sum(np.clip(MAR-trade_ror, 0., math.inf)>0)
    downside_deviation = np.sqrt(tmp1)
    DDR = APR / downside_deviation #Sortino

    metrics = {
        'APR': APR,
        'AVOL': AVOL,
        'ASR': ASR,
        'MDD': MDD,
        'CR': CR,
        'DDR': DDR
    }

    return metrics

def calculate_portfolio_metrics_with_pred_len(portfolio_values, pred_len, total_periods, risk_free_rate=0.01):
    """
    Calculate common portfolio metrics for performance evaluation,
    adjusting for custom pred_len-based periods.
    """
    # 연간화 요인 계산
    annual_factor = total_periods / pred_len

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