"""
# Application of ARMA model on real data

The aim of this project is to modelize the return of a stock using an ARMA model by following Box & Jenkins' methodology.

## I. Importation : packages & data
"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import statsmodels.api as sm
import statsmodels.tsa.api as smt
import scipy.stats as scs

fname = "CAC40.csv"
stock = pd.read_csv("data/{fname}".format(fname=fname), sep=";")
stock["Date"] = pd.to_datetime(stock["Date"], format="%d/%m/%Y")
stock = stock.set_index("Date")
stock.head()

"""## II. Stationarity and identification of the different possible models

Compute log-return and simple & partial autocorrelograms.
"""

# Log-return

# drop NaN, diff for derivation
rstock = np.log(stock["Close"]).diff().dropna()
rstock.plot()

# Simple & partial autocorrelograms
fig = plt.figure(figsize=(15, 6))
layout = (1, 2)
ax1 = plt.subplot2grid(layout, (0, 0))
ax2 = plt.subplot2grid(layout, (0, 1))
smt.graphics.plot_acf(rstock, ax=ax1)
smt.graphics.plot_pacf(rstock, ax=ax2)

"""Unit root test (enhanced Dickey-Fuller) on the return time series."""

from statsmodels.tsa.stattools import adfuller
adfuller(rstock)

"""H1 : is the time series stationary

p-value = 0.0, probability of being lower than the test statistic. Here, we are lower than the 1%, 5% and 10% significance level threshold.

We reject the null hypothesis H0, there is no unit root so the time series in stationary.

If the time series isn't stationary, one needs to transform it, else we can directly fit an ARMA model on it. We use autocorrelograms to find the model that we want to test.

## III. Model estimation and selection of the best one

Estimate the selected models and choose the best one accoring to the significativity on the coefficients and information criteria : we use the `ARMA` function of `statsmodels.tsa.api`.
"""

model = smt.ARMA(rstock.to_period('D'), order=(1, 0), missing="drop").fit(trend="nc")  # AR(1)
# trend="nc" to remove the mean value
model.summary()

# Test ARMA(p, q) models
for p in range(10):
  for q in range(10):
    if p != 0 and q != 0:
      try:
        model = smt.ARMA(rstock.to_period('D'), order=(p, q), missing="drop").fit(trend="nc")
        print("(" + str(p) + ", " + str(q) + ")    " + str(model.aic) + ",   " + str(model.pvalues["ar.L" + str(p) + ".Close"]) + ",   " + str(model.pvalues["ma.L" + str(q) + ".Close"]))
      except ValueError:
        ()
        # print("(" + str(p) + ", " + str(q) + ") coefficients are not stationary")

"""## IV. Best model : residuals analysis

Let's analyse the residuals of the model we have selected : autocorrelograms, Ljung-Box test, histogram, qq-plot, normality test (Jarque-Bera, Shapiro-Wilk).
"""

residuals = model.resid
residuals = residuals / np.std(residuals)
residuals.plot()

# Autocorrelograms
fig = plt.figure(figsize=(15, 6))
layout = (1, 2)
ax1 = plt.subplot2grid(layout, (0, 0))
ax2 = plt.subplot2grid(layout, (0, 1))
smt.graphics.plot_acf(residuals, ax=ax1)
smt.graphics.plot_pacf(residuals, ax=ax2)

"""Ljung-Box"""

LB = sm.stats.diagnostic.acorr_ljungbox(residuals, lags=20)
plt.plot(range(1, 21, 1), LB[1], 'o')  # print p-values
plt.plot(range(1, 21, 1), 0.05 * np.ones(20))

"""Nothing below 5% for the significativity of the autocorrelograms : it seems to be a white noise."""

import scipy

mu = 0
sigma = 1
# Histogram
n, bins, patches = plt.hist(residuals, 100, facecolor='blue', alpha=0.75, density=True)

# Gaussian fit
y = scipy.stats.norm.pdf(bins, mu, sigma)
l = plt.plot(bins, y, 'r--', linewidth=2)

# Plot
plt.xlabel('log-return')
plt.ylabel('Density')
plt.title("Are residuals gaussians ?")
plt.show()

"""## V. Predictions

Calculate predictions and confidence intervals at 95% and 99%. Print the predictions and the invervals.
"""

model.predict()  # on the data

pred, err95, IC95 = model.forecast(20)
# pred : 20 days horizon forecast
# err95 : error at 95%
# IC95 : confidence intervals
pred

# for 99%, change alpha parameter
pred, err99, IC99 = model.forecast(20, alpha=0.01)
pred
