[README.md](https://github.com/user-attachments/files/26903096/README.md)
# ACC102 Track 4: Multi-Stock Interaction Dashboard

## Project overview

This project is an interactive Streamlit data product for ACC102 Track 4. It helps a retail investor compare how major US technology stocks move together, where diversification is still possible, and which stocks carry higher risk.

The dashboard focuses on seven stocks: `AAPL`, `MSFT`, `NVDA`, `AMZN`, `GOOGL`, `META`, and `TSLA`.

## Analytical problem

Retail investors often compare stock performance one ticker at a time, but portfolio decisions depend on interaction as well as individual returns. The analytical problem in this project is:

**How can a retail investor compare co-movement, volatility, and short lead-lag behaviour across major technology stocks to support better watchlist and diversification decisions?**

## Intended user

The main user is a retail investor or business student who wants a fast visual tool for comparing several large-cap stocks without reading raw spreadsheets or code outputs.

## Data source

- Source: Yahoo Finance via `yfinance`
- Access date: `2026-04-15`
- Coverage in the current dataset: `2022-01-03` to `2026-04-14`
- Variables used: adjusted open, high, low, close, and volume

## Python workflow

The project includes a substantive Python workflow rather than a presentation-only app:

1. Download daily stock data with `yfinance`
2. Clean and reshape the multi-ticker dataset with `pandas`
3. Calculate daily returns, indexed prices, rolling volatility, and drawdowns
4. Measure pairwise correlation and rolling correlation
5. Estimate beta against a chosen benchmark stock
6. Scan for simple lead-lag patterns over a short horizon
7. Present the results in an interactive Streamlit dashboard

## Main features

- Indexed price comparison
- Risk-return bubble chart
- Correlation heatmap
- Rolling correlation explorer
- Lead-lag comparison table
- Drawdown view
- Downloaded local dataset for reproducibility

## Project structure

- `app.py`: Streamlit dashboard
- `src/stock_analysis.py`: data cleaning and analysis logic
- `scripts/fetch_stock_data.py`: dataset download script
- `data/raw/us_tech_stocks.csv`: cached dataset used by the app
- `notebooks/stock_interaction_workflow.ipynb`: analytical workflow notebook
- `docs/reflection_report.md`: 500-800 word reflection report
- `docs/demo_video_script.md`: 1-3 minute demo narration script
- `docs/submission_checklist.md`: final upload checklist

## How to run locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Refresh the dataset if needed:

```bash
python3 scripts/fetch_stock_data.py --symbols AAPL MSFT NVDA AMZN GOOGL META TSLA --start 2022-01-01
```

Start the dashboard:

```bash
streamlit run app.py
```

## Assignment deliverables included

- Interactive Python tool
- Python notebook
- README
- Reflection report
- Demo video script

## Notes for publishing

To obtain the required product link for submission, push this project to GitHub and deploy `app.py` on Streamlit Community Cloud or another platform that supports Streamlit apps. The codebase is already structured for that workflow.
