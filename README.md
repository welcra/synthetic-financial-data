# Synthetic Financial Data Generation

The goal of this project is to provide realistic synthetic financial data for people to use when backtesting algorithms, or for whatever use cases anyone could have

Methodology:
The overall market OHLCV is first generated using a combination of different models. The data is generated through geometric brownian motion, with merton's jump diffusion model and a stochastic volatility model (heston). Data for individual stocks is then generated based on a predetermined alpha and beta, with ideosyncratic risk added after. Meanwhile, fundamentals are created based on predetermined "genes" of three factors: the growth score, quality score, and leverege score. The genes are dynamically based on stock performance to add to the realism of the data.

Next Step:
Now that I've finished adding fundamentals, my goal is to make sure the data is plausible and could work well for backtests