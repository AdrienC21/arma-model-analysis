# Application of ARMA model on real data

ARMA model and statistical analysis on real financial data.

## Introduction

The aim of this project is to model the return of a stock/index using an ARMA model by following Box & Jenkins' methodology.

All the functions have been applied to the CAC40 index and are contained in the file main.py

You will find in this README file below the code with the corresponding plots (as it if was a notebook).

## Installation

Clone this repository :

```bash
git clone https://github.com/AdrienC21/arma-model-analysis.git
```

or just follow the guide.

## I. Importation : packages & data

```python
import numpy as np
import pandas as pd
%matplotlib inline
import matplotlib.pylab as plt
import statsmodels.api as sm
import statsmodels.tsa.api as smt
import scipy.stats as scs
```

```python
fname = "CAC40.csv"
stock = pd.read_csv("data/{fname}".format(fname=fname), sep=";")
stock["Date"] = pd.to_datetime(stock["Date"], format="%d/%m/%Y")
stock = stock.set_index("Date")
stock.head()
```

![alt text](https://github.com/AdrienC21/arma-model-analysis/blob/main/images/dataframe.png?raw=true)

## II. Stationarity and identification of the different possible models

Compute log-return and simple & partial autocorrelograms.

```python
# Log-return
rstock = np.log(stock["Close"]).diff().dropna()  # drop NaN, diff for derivation
rstock.plot()
```

![alt text](https://github.com/AdrienC21/arma-model-analysis/blob/main/images/logreturn.png?raw=true)

```python
# Simple & partial autocorrelograms
fig = plt.figure(figsize=(15, 6))
layout = (1, 2)
ax1 = plt.subplot2grid(layout, (0, 0))
ax2 = plt.subplot2grid(layout, (0, 1))
smt.graphics.plot_acf(rstock, ax=ax1)
smt.graphics.plot_pacf(rstock, ax=ax2)
```

![alt text](https://github.com/AdrienC21/arma-model-analysis/blob/main/images/autocorrel1.png?raw=true)

Unit root test (enhanced Dickey-Fuller) on the return time series.

```python
from statsmodels.tsa.stattools import adfuller
adfuller(rstock)
```

(-24.10192790416603,
 0.0,
 4,
 2293,
 {'1%': -3.4332050526159112,
  '10%': -2.5674414457185817,
  '5%': -2.8628012970364574},
 -13454.299858834613)

**H1** : is the time series stationary

p-value = 0.0, probability of being lower than the test statistic. Here, we are lower than the 1%, 5% and 10% significance level threshold.

We reject the null hypothesis **H0**, there is no unit root so the time series in stationary.

If the time series isn't stationary, one needs to transform it, else we can directly fit an ARMA model on it. We use autocorrelograms to find the model that we want to test.

## III. Model estimation and selection of the best one

Estimate the selected models and choose the best one accoring to the significativity on the coefficients and information criteria : we use the `ARMA` function of `statsmodels.tsa.api`.

```python
model = smt.ARMA(rstock.to_period('D'), order=(1, 0), missing="drop").fit(trend="nc")  # AR(1)
# trend="nc" to remove the mean value
model.summary()
```
![alt text](https://github.com/AdrienC21/arma-model-analysis/blob/main/images/ARMAmodelresult.png?raw=true)

```python
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

"""
(1, 7)    -13614.974997061401,   0.06487765416345079,   0.23445909458780612
(1, 8)    -13614.605494934867,   0.012147270256365366,   0.12885729490459125
(1, 9)    -13612.746085008564,   0.08320971253940523,   0.6988569200633201
...
"""
```

## IV. Best model : residuals analysis

Let's analyse the residuals of the model we have selected : autocorrelograms, Ljung-Box test, histogram, qq-plot, normality test (Jarque-Bera, Shapiro-Wilk).

```python
residuals = model.resid
residuals = residuals / np.std(residuals)
residuals.plot()
```

![alt text](https://github.com/AdrienC21/arma-model-analysis/blob/main/images/resid.png?raw=true)

```python
# Autocorrelograms
fig = plt.figure(figsize=(15, 6))
layout = (1, 2)
ax1 = plt.subplot2grid(layout, (0, 0))
ax2 = plt.subplot2grid(layout, (0, 1))
smt.graphics.plot_acf(residuals, ax=ax1)
smt.graphics.plot_pacf(residuals, ax=ax2)
```

![alt text](https://github.com/AdrienC21/arma-model-analysis/blob/main/images/residautocorr.png?raw=true)

Ljung-Box

```python
LB = sm.stats.diagnostic.acorr_ljungbox(residuals, lags=20)
plt.plot(range(1, 21, 1), LB[1], 'o')  # print p-values
plt.plot(range(1, 21, 1), 0.05 * np.ones(20))
```

![alt text](https://github.com/AdrienC21/arma-model-analysis/blob/main/images/ljungbox.png?raw=true)

Nothing below 5% for the significativity of the autocorrelograms : it seems to be a white noise.

```python
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
```

![alt text](https://github.com/AdrienC21/arma-model-analysis/blob/main/images/gauss.png?raw=true)

## V. Predictions

Calculate predictions and confidence intervals at 95% and 99%. Print the predictions and the invervals.

```python
model.predict()  # on the data
```

![alt text](https://github.com/AdrienC21/arma-model-analysis/blob/main/images/f1.png?raw=true)

```python
pred, err95, IC95 = model.forecast(20)
# pred : 20 days horizon forecast
# err95 : error at 95%
# IC95 : confidence intervals
pred
```

![alt text](https://github.com/AdrienC21/arma-model-analysis/blob/main/images/f2.png?raw=true)

```python
# for 99%, change alpha parameter
pred, err99, IC99 = model.forecast(20, alpha=0.01)
pred
```

![alt text](https://github.com/AdrienC21/arma-model-analysis/blob/main/images/f3.png?raw=true)
## License

[MIT](https://choosealicense.com/licenses/mit/)
