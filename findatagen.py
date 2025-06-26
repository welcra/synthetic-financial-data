import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

T = 50
dt = 1/252
N = int(T / dt)
dates = pd.bdate_range(end=pd.Timestamp.today(), periods=N)

S0 = 1
mu = 0.06
v0 = 0.04
kappa = 2.0
theta = 0.04
xi = 0.3

lambda_jump = 0.1
mu_jump = -0.02
sigma_jump = 0.1

S = np.zeros(N)
v = np.zeros(N)
r = np.zeros(N)
jump_occurred = np.zeros(N)

S[0] = S0
v[0] = v0

for t in range(1, N):
    Z1 = np.random.normal()
    Z2 = np.random.normal()
    dW_s = Z1
    dW_v = 0.7 * Z1 + np.sqrt(1 - 0.7 ** 2) * Z2

    dN = np.random.poisson(lambda_jump * dt)
    J = np.random.normal(mu_jump, sigma_jump) if dN > 0 else 0
    jump_occurred[t] = dN

    v[t] = np.abs(
        v[t-1] + kappa * (theta - v[t-1]) * dt + xi * np.sqrt(v[t-1]) * dW_v * np.sqrt(dt)
    )

    r[t] = (mu - 0.5 * v[t]) * dt + np.sqrt(v[t]) * dW_s * np.sqrt(dt) + J * dN
    S[t] = S[t-1] * np.exp(r[t])

high_noise = np.random.uniform(0.001, 0.02, N)
low_noise = np.random.uniform(0.001, 0.02, N)
open_ = np.roll(S, 1)
open_[0] = S0
high = S * (1 + high_noise)
low = S * (1 - low_noise)
close = S
volume = np.random.normal(1e6, 1e5, N).astype(int)

df = pd.DataFrame({
    "Date": dates,
    "Open": open_,
    "High": high,
    "Low": low,
    "Close": close,
    "Volume": volume,
    "Volatility": np.sqrt(v),
    "Jump": jump_occurred
})
df.to_csv(f"mkt/mktgen.csv", index=False)

S0_stocks = np.random.uniform(20, 300, 1400)

betas = np.random.normal(1.0, 0.4, 1400)

alphas = np.random.normal(0.0, 0.01, 1400)

sigmas_idio = np.random.uniform(0.20, 0.50, 1400)

market_log_returns_col = r.reshape(-1, 1)

idio_shocks = np.random.normal(0, 1, size=(N, 1400)) * sigmas_idio * np.sqrt(dt)

log_returns_stocks = alphas * dt + betas * market_log_returns_col + idio_shocks

log_returns_stocks[0, :] = 0

cumulative_log_returns = np.cumsum(log_returns_stocks, axis=0)

S_stocks = S0_stocks * np.exp(cumulative_log_returns)

close_stocks = S_stocks

open_stocks = np.roll(close_stocks, 1, axis=0)
open_stocks[0, :] = S0_stocks

high_noise = np.random.uniform(0.00, 0.015, size=(N, 1400))
low_noise = np.random.uniform(0.00, 0.015, size=(N, 1400))

high_stocks = close_stocks * (1 + high_noise)
low_stocks = close_stocks * (1 - low_noise)

high_stocks = np.maximum(high_stocks, np.maximum(open_stocks, close_stocks))
low_stocks = np.minimum(low_stocks, np.minimum(open_stocks, close_stocks))


volume_stocks = np.random.normal(5e5, 8e4, size=(N, 1400)).astype(int)
volume_stocks[volume_stocks < 0] = 0

all_stock_data = {}
for i in range(1400):
    ticker = f"STOCK_{i+1}"
    df_stock = pd.DataFrame({
        "Date": dates,
        "Open": open_stocks[:, i],
        "High": high_stocks[:, i],
        "Low": low_stocks[:, i],
        "Close": close_stocks[:, i],
        "Volume": volume_stocks[:, i]
    })
    all_stock_data[ticker] = df_stock

for ticker, df in all_stock_data.items():
    df.to_csv(f"stock_data/{ticker}.csv", index=False)