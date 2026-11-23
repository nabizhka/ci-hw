# VaR confidence intervals for $1M of AAPL (MF731 HW 13, sheet `13_CI_pub`)

Closes 11/1/2016 to 10/31/2018 (504 closes, 503 log returns), frozen in `data/`. alpha = .97, beta = .02, M = 125,000, seed 731.

(a) empirical 24,984.39; formula (1) 22,773.55, 98% CI (21,138.53, 24,651.99).
(b) mean 22,808.77, interval (20,698.11, 25,091.16).

```bash
pip install -r requirements.txt
python src/main.py        # ~3 s, asserts the checks, writes generated/ and figures/
latexmk -pdf main.tex
```
`src/get_data.py` re-downloads the data; it is not needed to reproduce the report.
