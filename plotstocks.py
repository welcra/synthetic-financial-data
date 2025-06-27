import matplotlib.pyplot as plt
import pandas as pd
import random

stocks = [random.randint(1, 1401) for _ in range(20)]

for i in stocks:
    plt.figure(i)
    data = pd.read_csv(f"stock_data/STOCK_{i}.csv")
    fund = pd.read_csv(f"fundamentals/STOCK_{i}.csv")

    data["Date"] = pd.to_datetime(data["Date"], errors='raise')
    fund["Date"] = pd.to_datetime(fund["Date"], errors='raise')

    data = data.sort_values("Date")
    fund = fund.sort_values("Date")
    
    merged = pd.merge_asof(data, fund[["Date", "eps"]], on="Date", direction="backward")
    plt.plot(merged["eps"], merged["Close"], label=f'STOCK_{i}')

plt.show()