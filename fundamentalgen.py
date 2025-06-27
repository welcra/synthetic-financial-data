import pandas as pd
import numpy as np
import os

df = pd.read_csv("mkt/mktgen.csv")
all_stock_data = {}

for i in os.listdir("stock_data"):
    all_stock_data[i[0:-4]] = pd.read_csv(f"stock_data/{i}")
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)
quarter_dates = pd.date_range(start=df.index[0], end=df.index[-1], freq="Q")

num_quarters = len(quarter_dates)
num_stocks = 1400

def bounded_normal(mean, std, low, high, size):
    mean = np.asarray(mean)
    std = np.broadcast_to(std, mean.shape)
    return np.clip(np.random.normal(mean, std), low, high)

growth_score = np.random.uniform(-1, 1, num_stocks)
quality_score = np.random.uniform(-1, 1, num_stocks)
leverage_score = np.random.uniform(-1, 1, num_stocks)

fundamentals_dict = {}

for i in range(num_stocks):
    ticker = f"STOCK_{i+1}"
    
    prices = all_stock_data[ticker]["Close"].values
    quarterly_prices = prices[np.searchsorted(df.index.values, quarter_dates.values)]
    quarterly_prices = np.asarray(quarterly_prices).flatten()
    returns = np.diff(np.log(quarterly_prices), prepend=np.log(quarterly_prices[0]))

    score_growth = growth_score[i] + returns
    score_quality = quality_score[i] + returns
    score_leverage = leverage_score[i] - returns

    eps = bounded_normal(2 + 3 * score_growth, 0.5, -5, 10, num_quarters)
    ebit_per_share = eps + bounded_normal(1, 0.2, 0, 3, num_quarters)

    book_value = bounded_normal(30 + 10 * score_quality, 5, 5, 100, num_quarters)
    cash_ratio = bounded_normal(0.5 + 0.2 * score_quality, 0.1, 0.1, 1.5, num_quarters)
    current_ratio = cash_ratio + bounded_normal(0.8, 0.2, 0.5, 3, num_quarters)
    quick_ratio = current_ratio - bounded_normal(0.3, 0.1, 0.1, 1, num_quarters)

    gross_margin = bounded_normal(0.6 + 0.05 * score_quality, 0.05, 0.2, 0.9, num_quarters)
    operating_margin = gross_margin - bounded_normal(0.1, 0.02, 0.05, 0.4, num_quarters)
    net_margin = operating_margin - bounded_normal(0.05, 0.01, 0.01, 0.3, num_quarters)
    pretax_margin = net_margin + bounded_normal(0.02, 0.01, 0.01, 0.1, num_quarters)

    fcf_margin = bounded_normal(net_margin, 0.03, 0, 0.4, num_quarters)
    sga_to_sale = bounded_normal(0.1 + 0.05 * (1 - score_quality), 0.02, 0.05, 0.3, num_quarters)
    sales_per_share = eps / np.maximum(net_margin, 0.01)

    roa = bounded_normal(0.05 + 0.05 * score_quality, 0.02, -0.1, 0.3, num_quarters)
    roe = bounded_normal(0.1 + 0.1 * score_quality, 0.03, -0.2, 0.5, num_quarters)
    roic = bounded_normal(0.08 + 0.08 * score_quality, 0.02, -0.1, 0.4, num_quarters)
    rotc = bounded_normal(roic - 0.01, 0.01, -0.1, 0.3, num_quarters)

    longterm_debt_total_asset = bounded_normal(0.3 + 0.2 * score_leverage, 0.05, 0, 0.9, num_quarters)
    longterm_debt_total_capital = longterm_debt_total_asset + bounded_normal(0.05, 0.01, 0, 0.1, num_quarters)
    longterm_debt_total_equity = longterm_debt_total_capital + bounded_normal(0.05, 0.02, 0, 0.2, num_quarters)

    total_debt_to_equity = longterm_debt_total_equity + bounded_normal(0.1, 0.05, 0, 1.5, num_quarters)
    total_debt_to_total_asset = longterm_debt_total_asset + bounded_normal(0.05, 0.02, 0, 1.0, num_quarters)
    total_debt_to_total_capital = longterm_debt_total_capital + bounded_normal(0.05, 0.02, 0, 1.0, num_quarters)

    net_debt_to_total_capital = total_debt_to_total_capital - bounded_normal(0.05, 0.01, 0, 0.5, num_quarters)
    net_debt_to_total_equity = total_debt_to_equity - bounded_normal(0.05, 0.01, 0, 0.5, num_quarters)

    df_fund = pd.DataFrame({
        "Date": quarter_dates,
        "bookValue": book_value,
        "cashRatio": cash_ratio,
        "currentRatio": current_ratio,
        "ebitPerShare": ebit_per_share,
        "eps": eps,
        "fcfMargin": fcf_margin,
        "grossMargin": gross_margin,
        "longtermDebtTotalAsset": longterm_debt_total_asset,
        "longtermDebtTotalCapital": longterm_debt_total_capital,
        "longtermDebtTotalEquity": longterm_debt_total_equity,
        "netDebtToTotalCapital": net_debt_to_total_capital,
        "netDebtToTotalEquity": net_debt_to_total_equity,
        "netMargin": net_margin,
        "operatingMargin": operating_margin,
        "pretaxMargin": pretax_margin,
        "quickRatio": quick_ratio,
        "roaTTM": roa,
        "roeTTM": roe,
        "roicTTM": roic,
        "rotcTTM": rotc,
        "salesPerShare": sales_per_share,
        "sgaToSale": sga_to_sale,
        "totalDebtToEquity": total_debt_to_equity,
        "totalDebtToTotalAsset": total_debt_to_total_asset,
        "totalDebtToTotalCapital": total_debt_to_total_capital,
        "totalRatio": current_ratio + quick_ratio + cash_ratio
    })

    df_fund.to_csv(f"fundamentals/{ticker}.csv", index=False)