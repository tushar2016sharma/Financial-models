# run_etl_pipeline.py

import os
import numpy as np
from datetime import date
import pandas as pd
from ETL import get_dow30_tickers, fetch_price_data, get_risk_free_rate, compute_monthly_returns, compute_var

def main():
    start_date = date(2010, 1, 1)
    end_date = date.today()

    tickers = get_dow30_tickers()
    price_df = fetch_price_data(tickers, start_date, end_date)
    rf_rate = get_risk_free_rate(start_date, end_date)

    # save monthly returns
    monthly_rets = compute_monthly_returns(price_df)
    os.makedirs("output", exist_ok=True)
    monthly_rets.to_csv("output/monthly_rets.csv")

    # save equally-weighted daily log returns
    lnret = np.log(price_df / price_df.shift(1)).dropna()
    ew = pd.Series(1.0 / len(price_df.columns), index=price_df.columns)
    hist_ret_ew = (lnret * ew).sum(axis=1)
    hist_ret_ew.to_csv("output/hist_ret_ew.csv", header=["returns"])

if __name__ == "__main__":
    main()