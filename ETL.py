import os
import numpy as np
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from datetime import date
import pandas_datareader as pdr
import pandas_datareader.data as web
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
from sqlalchemy import create_engine
from typing import Optional, List, Tuple, Dict



######################### EXTRACT #########################

def get_dow30_tickers() -> list[str]:
    # web scraping
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.82 Safari/537.36'}
    url = 'https://bullishbears.com/dow-jones-stocks-list/'
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, 'html.parser')
    table = soup.find('table')
    tickers = []
    for row in table.find_all('tr')[1:]:
        cols = row.find_all('td')
        if not cols:
            continue
        ticker = cols[0].get_text(strip=True)
        tickers.append(ticker)
    return tickers


def fetch_price_data(tickers: list[str], start_date: date, end_date: date, interval: str = "1d") -> pd.DataFrame:
    dailyprc_df = pd.DataFrame()
    for tkr in tickers:
        df = yf.download(tkr, start=start_date, end=end_date, interval=interval, progress=False)
        if df.empty:
            continue
        #dailyprc_df[tkr] = df["Adj Close"]

        # pick adjusted close if available, otherwise fall back to close
        if "Adj Close" in df.columns:
            series = df["Adj Close"]
        elif "Close" in df.columns:
            series = df["Close"]
        else:
            continue

        dailyprc_df[tkr] = series

    # if there's a 'DOW' column (with sparse data), drop it:
    return dailyprc_df.drop(columns=["DOW"], errors="ignore")


def get_risk_free_rate(start_date: date, end_date: date, fred_symbol: str = "DGS10") -> float:
    allyr_yield = pdr.DataReader(fred_symbol, 'fred', start_date, end_date).dropna()
    rf = allyr_yield.iloc[0][0] / 100
    return float(rf)



######################### TRANSFORM #########################

def compute_monthly_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    monthly_prices = price_df.resample('ME').last()
    monthly_returns = monthly_prices.pct_change().dropna()
    return monthly_returns


def sliding_windows(start_year: int, end_year: int, window_size: int = 3) -> list[tuple[int,int]]:
    return [(y, y + window_size - 1) for y in range(start_year, end_year - window_size + 2)]


def simulate_minvar_maxsharpe(window_data: pd.DataFrame, rf: float, num_sim: int = 50_000) -> tuple[pd.Series, pd.Series]:
    # annualize mean and covariance
    mu_vec = np.array(window_data.mean() * 12)
    cov_mat = np.array(window_data.cov() * 12)
    N = window_data.shape[1]

    port_expret = np.empty(num_sim)
    port_var = np.empty(num_sim)
    weights = np.empty((num_sim, N))

    for i in range(num_sim):
        rnd = np.random.rand(N)
        w = rnd / rnd.sum()
        weights[i] = w
        port_expret[i] = w.dot(mu_vec)
        port_var[i] = w.dot(cov_mat).dot(w)

    port_sd = np.sqrt(port_var)
    # minimum-variance portfolio:
    min_idx = port_sd.argmin()
    min_var_weights = weights[min_idx]

    # maximum Sharpe ratio:
    sharpes = (port_expret - rf) / port_sd
    max_idx = sharpes.argmax()
    max_sharpe_weights = weights[max_idx]

    return (
        pd.Series(min_var_weights, index=window_data.columns),
        pd.Series(max_sharpe_weights, index=window_data.columns)
    )


def compute_analytical_max_sharpe(window_data: pd.DataFrame, rf: float) -> pd.Series:
    mu_vec = np.array(window_data.mean() * 12)
    cov_mat = np.array(window_data.cov() * 12)
    Sinv = np.linalg.inv(cov_mat)
    diff = mu_vec - rf
    top = Sinv.dot(diff)
    denom = np.ones(len(diff)).dot(Sinv).dot(diff)
    w_star = top / denom
    return pd.Series(w_star, index=window_data.columns)


def compute_optimized_max_sharpe(window_data: pd.DataFrame, rf: float, bounds: Optional[List[Tuple[float, float]]] = None) -> pd.Series:
    mu_vec = np.array(window_data.mean() * 12)
    cov_mat = np.array(window_data.cov() * 12)
    N = len(mu_vec)

    def port_sd(w: np.ndarray) -> float:
        return np.sqrt(w.dot(cov_mat).dot(w))

    def neg_sharpe(w: np.ndarray) -> float:
        return -((w.dot(mu_vec) - rf) / port_sd(w))

    cons = ({'type': 'eq', 'fun': lambda w: w.sum() - 1})
    if bounds is None:
        bounds = [(0.0, 1.0)] * N

    x0 = np.ones(N) / N
    result = minimize(neg_sharpe, x0, method='SLSQP', bounds=bounds, constraints=cons)
    return pd.Series(result.x, index=window_data.columns)


def compute_analytical_gmv(window_data: pd.DataFrame) -> pd.Series:
    cov_mat = np.array(window_data.cov() * 12)  # annualized covariance
    N = cov_mat.shape[0]
    TwoS = 2 * cov_mat

    # Build A matrix: top N rows: [2Σ | 1], last row: [1^T | 0]
    A = np.zeros((N + 1, N + 1))
    A[:N, :N] = TwoS
    A[:N, N] = 1.0
    A[N, :N] = 1.0
    # b vector is zeros for first N entries, then 1
    b = np.zeros(N + 1)
    b[N] = 1.0

    sol = np.linalg.inv(A).dot(b)
    w = sol[:N]
    return pd.Series(w, index=window_data.columns)


def compute_var(hist_returns: pd.Series, days: int, thresholds: list[float],
                portfolio_value: float = 100.0, num_sim: int = 50_000) -> Tuple[Dict[str, List[float]], pd.Series, np.ndarray]:
    # compute mean and std of daily returns
    mu = hist_returns.mean()
    sigma = hist_returns.std()

    # rolling sum of days log‐returns
    hist_Nday = hist_returns.rolling(days).sum().dropna()
    hist_Nday_dollars = hist_Nday * portfolio_value

    Var_dict = {"Param": [], "Hist": [], "Monte": []}

    # parametric VaR
    for alpha in thresholds:
        z = norm.ppf(alpha)
        var_pct = mu * days - abs(z) * sigma * np.sqrt(days)
        Var_dict["Param"].append(var_pct * portfolio_value)

    # historical VaR
    for alpha in thresholds:
        VaR_hist = np.nanpercentile(hist_Nday_dollars, alpha * 100)
        Var_dict["Hist"].append(VaR_hist)

    # monte carlo VaR
    np.random.seed(100)
    zs = np.random.normal(0, 1, size=num_sim)
    mc_returns = (mu * days + zs * sigma * np.sqrt(days)) * portfolio_value

    for alpha in thresholds:
        var_monte = np.nanpercentile(mc_returns, alpha * 100)
        Var_dict["Monte"].append(var_monte)

    # hist_Nday_dollars is the empirical rolling returns
    # simulated_losses is the monte carlo array
    return Var_dict, hist_Nday_dollars, mc_returns



######################### LOAD #########################

def save_to_csv(df: pd.DataFrame, filepath: str, index: bool = True, **kwargs) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=index, **kwargs)


def save_to_parquet(df: pd.DataFrame, filepath: str, index: bool = True, **kwargs) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_parquet(filepath, index=index, **kwargs)


def save_to_database(df: pd.DataFrame, table_name: str, db_url: str, if_exists: str = "replace") -> None:
    engine = create_engine(db_url)
    df.to_sql(table_name, engine, if_exists=if_exists, index=True)
    engine.dispose()


def save_portfolio_weights_dict(weights_dict: dict[str, dict[str, pd.Series]], base_path: str, format: str = "csv") -> None:
    os.makedirs(base_path, exist_ok=True)
    for window, methods in weights_dict.items():
        win_dir = os.path.join(base_path, window)
        os.makedirs(win_dir, exist_ok=True)
        for method_name, series in methods.items():
            fname = f"{method_name}.{format}"
            fullpath = os.path.join(win_dir, fname)
            if format.lower() == "csv":
                series.to_csv(fullpath, header=True)
            elif format.lower() == "parquet":
                series.to_frame().to_parquet(fullpath, index=True)
            else:
                raise ValueError(f"Unsupported format: {format}")


def save_var_dict(var_dict: dict[str, list[float]], filepath: str) -> None:
    thresholds = sorted(var_dict["Param"], key=lambda _: _)  # we assume aligned order
    df = pd.DataFrame(var_dict, index=[f"{int(p*100)}%" for p in var_dict["Param"]])
    df.index.name = "Threshold"
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath)