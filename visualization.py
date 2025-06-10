import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def plot_top10_stock_weights(weight_df: pd.DataFrame, title: str) -> Figure:
    top_10 = weight_df.index[:10]
    fig, ax = plt.subplots(figsize=(8, 4))
    # portfolio analysis plot for 10 tickers
    for ticker in top_10:
        ax.plot(weight_df.columns, weight_df.loc[ticker], marker = 'o', 
                markersize = 2, linewidth = 0.5, label=ticker)

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Year", fontsize=10)
    ax.set_ylabel("Stock weights", fontsize=10)
    ax.legend(loc="upper right", fontsize=5, ncol=2)
    plt.setp(ax.get_xticklabels(), rotation=45)
    fig.tight_layout()
    return fig


def plot_var_distribution(hist_Nday_ret_dollars: pd.Series, var_dict: dict[str, list[float]], days: int) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(hist_Nday_ret_dollars.dropna(), bins=50, density=True, alpha=0.5, label=f"{days}-Day Returns")
    # parametric VaR lines
    colors = ["purple", "black", "red"]
    for pct, raw, c in zip([10, 5, 1], var_dict["Param"], colors):
        x = float(raw)
        # make sure this is a Python float, not a series or array:
        try:
            x = float(raw)
        except Exception:
            # if it’s a 1-element array or series, grab its first element:
            if hasattr(raw, 'iloc'):
                x = raw.iloc[0]
            else:
                x = raw[0]
        ax.axvline(x, linestyle="--", color=c, label=f"{pct}% VaR for {days} days")

    ax.set_xlabel(f"{days}-Day Portfolio Return ($)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Distribution of Portfolio {days}-Day Returns and Parametric VaR")
    ax.legend(fontsize = 5)
    fig.tight_layout()
    return fig


def plot_historical_var(hist_vals: pd.Series, var_dict: dict[str, list[float]], days: int) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(hist_vals, bins=50, density=True, alpha=0.5, label=f"{days}-Day Returns")
    # add threshold lines
    colors = ["purple", "black", "red"] 
    for pct, raw, c in zip([10, 5, 1], var_dict["Hist"], colors):
        x = float(raw)
        ax.axvline(x, linestyle="--", color=c, label=f"{pct}% VaR for {days} days")
        
    ax.set_title(f"VaR for {days}-Day Horizon (historical method)")
    ax.set_xlabel(f"{days}-Day Return ($)")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize = 5)
    fig.tight_layout()
    return fig


def plot_montecarlo_var(mc_vals: np.ndarray, var_dict: dict[str, list[float]], days: int) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(mc_vals, bins=50, density=True, alpha=0.5, label=f"{days}-Day Simulated Returns")
    # add threshold lines
    colors = ["purple", "black", "red"]
    for pct, raw, c in zip([10, 5, 1], var_dict["Monte"], colors):
        x = float(raw)
        ax.axvline(x, linestyle="--", color=c, label=f"{pct}% VaR for {days} days")
    ax.set_title(f"VaR for {days}-Day Horizon (Monte Carlo method)")

    ax.set_xlabel(f"{days}-Day Return ($)")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize = 5)
    fig.tight_layout()
    return fig