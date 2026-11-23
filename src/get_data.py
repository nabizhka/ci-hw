"""Download AAPL daily closes from Yahoo Finance once and freeze them in data/."""
import os
import yfinance as yf

ROOT = os.path.join(os.path.dirname(__file__), "..")
df = yf.download("AAPL", start="2016-11-01", end="2018-11-02", auto_adjust=False, progress=False)
out = df[["Close", "Adj Close"]].copy()
out.columns = ["Close", "AdjClose"]
out.index.name = "Date"
out.to_csv(os.path.join(ROOT, "data", "aapl_close.csv"))
print(len(out), out.index[0].date(), out.index[-1].date())
