# Financial-models
Portfolio analysis and value at risk analysis of Dow30 tickers

The comapnies analyzed are the 30 companies in the current Dow Jones index. An ETL pipeline is built which first extracts the realtime stock return data from yfinance API,
performs transformations (which include all the computations required for portfolio analysis and value at risk analysis) and finally provides loading options to different sources
as desired. The stock return data is taken from January 2010 till today's date. The output can be viewed on a Streamlit UI.

### Portfolio Analysis:

Based on the user-specified year window, the variation of the stock weights over time is displayed by following methods of portfolio analysis:

- a. Minimum variance portfolio in a simulation
- b. max Sharpe portfolio in a simulation
- c. max Sharpe portfolio analytically
- d. max Sharpe portfolio analytically with the constraint that each weight must be positive and no weight can be bigger than 10%
- e. Global Minimum Variance portfolio analytically

### Value at Risk Analysis:

The user specifies the time period and the portfolio amount. Then, the VaR analysis is conducted on an equally weighted portfolio of all stocks for the entire sample period, using these three methods:

- Parametric
- Historical
- Monte Carlo simulation

### Steps to run:

Clone the github repository

```
git clone https://github.com/tushar2016sharma/Financial-models.git
```

Navigate to repo directory, install the dependencies

```
cd /path/to/Financial-models

pip install -r requirements.txt
```

Run the driver and app files

```
python driver.py

streamlit run app.py
```


