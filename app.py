# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ETL import (get_risk_free_rate, get_dow30_tickers, fetch_price_data, sliding_windows, simulate_minvar_maxsharpe, 
                compute_analytical_max_sharpe, compute_optimized_max_sharpe, compute_analytical_gmv, compute_var)
from visualization import plot_top10_stock_weights, plot_var_distribution, plot_historical_var, plot_montecarlo_var


st.set_page_config(page_title="DOW30 Portfolio & Value at Risk", layout="wide")

@st.cache_data
def load_data():
    # load pre-computed files
    monthly_rets = pd.read_csv("output/monthly_rets.csv", index_col=0, parse_dates=True)
    hist_ret_ew_df = pd.read_csv("output/hist_ret_ew.csv", index_col=0)
    hist_ret_ew    = hist_ret_ew_df.iloc[:, 0]

    # also grab tickers & rf for portfolio logic
    tickers = list(monthly_rets.columns)
    rf_rate = get_risk_free_rate(monthly_rets.index.min().date(), monthly_rets.index.max().date())
    return monthly_rets, hist_ret_ew, tickers, rf_rate

monthly_rets, hist_ret_ew, tickers, rf = load_data()

st.title("📈 DOW30 Portfolio Analysis & Value at Risk Explorer 📉")

# ------------- Sidebar inputs -------------
with st.sidebar:
    # portfolio analysis
    st.header("Portfolio Analysis Settings")
    window_size = st.slider("Window size (years)", min_value=1, max_value=10, value=3)
    port_method = st.selectbox("Select portfolio method", ["Monte Carlo simulation", "Analytical", 
                                                           "Constrained optimization", "GMV (analytical)"])

    # VaR analysis
    st.markdown("---")
    st.header("VaR Settings")
    var_days = st.slider("VaR horizon (days)", min_value=1, max_value=60, value=30)
    portfolio_value = st.number_input("Portfolio Value ($)", value=100.0, step=10.0)
    var_method = st.selectbox("VaR Method", ["Parametric", "Historical", "Monte Carlo"])
    
    # download data
    st.markdown("---")
    st.header("📥 Download CSV data")
    monthly_csv = monthly_rets.to_csv().encode("utf-8")
    hist_csv    = hist_ret_ew.to_csv().encode("utf-8")

    st.download_button(label="Download Monthly Returns (CSV)", data=monthly_csv,
                       file_name="monthly_rets.csv", mime="text/csv")

    st.download_button(label="Download Daily Log Returns (CSV)", data=hist_csv,
                       file_name="hist_ret_ew.csv", mime="text/csv")
    
    # run analysis buttons
    st.markdown("---")
    run_port = st.button("Run Portfolio Analysis")
    run_var = st.button("Run VaR Analysis")


# ------------- Portfolio Analysis -------------
if run_port:
    # build sliding windows
    years = sliding_windows(start_year=monthly_rets.index.year.min(),
                            end_year=monthly_rets.index.year.max(),
                            window_size=window_size)

    # compute all weight-DataFrames
    port_dfs = {"Monte Carlo simulation": pd.DataFrame(index=tickers),
                "Analytical": pd.DataFrame(index=tickers),
                "Constrained optimization": pd.DataFrame(index=tickers),
                "GMV (analytical)": pd.DataFrame(index=tickers)}

    for (y0, y1) in years:
        label = f"{y0}-{y1}"
        wdata = (monthly_rets.loc[f"{y0}-01-01":f"{y1}-12-31"].dropna())

        # simulation
        minv, maxs = simulate_minvar_maxsharpe(wdata, rf)
        port_dfs["Monte Carlo simulation"][label] = maxs

        # analytical Sharpe
        port_dfs["Analytical"][label] = compute_analytical_max_sharpe(wdata, rf)

        # constrained
        port_dfs["Constrained optimization"][label] = compute_optimized_max_sharpe(wdata, rf, bounds=[(0.0, 0.1)] * wdata.shape[1])

        # GMV
        port_dfs["GMV (analytical)"][label] = compute_analytical_gmv(wdata)

    st.subheader(f"Top 10 weights over time — {port_method}")
    fig = plot_top10_stock_weights(port_dfs[port_method], title=port_method)
    st.pyplot(fig)


# ------------- VaR Analysis -------------
if run_var:
    # get both VaR percentiles and the full distributions
    var_dict, histN, mc_vals = compute_var(hist_returns=hist_ret_ew, days=var_days, thresholds=[0.10, 0.05, 0.01], 
                                           portfolio_value=portfolio_value, num_sim=50_000)

    st.subheader(f"{var_method} VaR for {var_days}-day Horizon")

    if var_method == "Parametric":
        fig = plot_var_distribution(histN, var_dict, var_days)
        st.pyplot(fig)

    elif var_method == "Historical":
        fig = plot_historical_var(histN, var_dict, var_days)
        st.pyplot(fig)

    else:  # Monte Carlo
        fig = plot_montecarlo_var(mc_vals, var_dict, var_days)
        st.pyplot(fig)