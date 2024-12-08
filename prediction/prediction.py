import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, kpss, InterpolationWarning, grangercausalitytests
from statsmodels.stats.stattools import durbin_watson
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.signal import detrend as scdetrend
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from sklearn.preprocessing import PowerTransformer

import warnings

warnings.filterwarnings('ignore', category=InterpolationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

stocks = pd.read_csv("../data-collection/stocks.csv", header=[0,1], index_col=[0])
ipos = pd.read_csv("../analysis/clustered.csv")

def adf_test(series):
    """ returns test statistic and p-value and lags """
    r = adfuller(series, regression='ct')
    return r[0], r[1], r[2]

def kpss_test(series):
    """ returns test statistic and p-value and lags """
    r = kpss(series, regression='ct')
    return r[0], r[1], r[2]

def durbinWatson_test(series):
    df=pd.DataFrame()
    df["1"] = series
    df["2"] = df["1"].shift(1)
    df.dropna(inplace=True)
    model = sm.OLS(df["1"], df["2"]).fit()
    residuals = model.resid
    r = durbin_watson(residuals)
    return r

def granger_test(series1, series2):
    data = pd.DataFrame({'series1': series1, 'series2': series2})
    test_result = grangercausalitytests(data[['series2', 'series1']], 5, verbose=False)
    p_values = {lag: float(test_result[lag][0]['ssr_ftest'][1]) for lag in range(1, 5 + 1)}
    return p_values

# group based on clusters
group2 = ipos[ipos["Cluster"] == 0]
group1 = ipos[ipos["Cluster"] == 1]

close = stocks.xs('Close', axis=1, level=1)
close.index = stocks.index

def getGroupClosingPrices(group, close):
    # separate just the closing data for each stock
    out = close.copy(deep=True)
    g1Symbols = [str(i) for i in group1["symbol"]]
    group1Stocks = [i for i in out.columns if str(i) in g1Symbols]
    out = out[group1Stocks]
    return out

g1StockData = getGroupClosingPrices(group1, close)

lambdas = {}
for c in g1StockData.columns:
    # p = PowerTransformer(method='box-cox')
    # g1StockData[c] = p.fit_transform(g1StockData[[c]])
    # lambdas[c] = p

    g1StockData[c] = g1StockData[c].apply(lambda x: np.log(x) if x != 0 else 0)
    # g1StockData[c] = g1StockData[c].diff()
    # g1StockData[c].dropna(inplace=True)

    ts, p, lags = adf_test(g1StockData[c].dropna())
    tsK, pK, lagsK = kpss_test(g1StockData[c].dropna())
    db = durbinWatson_test(g1StockData[c].dropna())
    print(c)
    print("Test Statistic:", ts, "\t", tsK)
    print("P-Value:       ", p, "\t", pK)
    print("Lags:          ", lags, "\t", lagsK)
    print("DurbinWatson: %.5f" % db)
    others = [x for x in g1StockData.columns if x != c]
    gc = {}
    print()
    for x in others:
        gc[x] = min(granger_test(g1StockData[c], g1StockData[x]).values())
        print(x, gc[x])
    print(gc)
    if p > 0.05 and pK > 0.05:
        g1StockData.drop(columns=c, inplace=True)
    

print(g1StockData)

for c in g1StockData.columns:
    plt.plot(g1StockData.index, g1StockData[c].apply(lambda x: np.exp(x) if x != 0 else 0), label=c)

# plt.legend()
plt.xticks(rotation=90)
plt.show()

g1StockData = g1StockData.iloc[:, 0:20]
date_labels = pd.date_range(start='2024-01-01', periods=60, freq='D')
g1StockData["date"] = date_labels
g1StockData.set_index("date", inplace=True)

model = VAR(g1StockData.iloc[:-5])
results = model.fit(maxlags=1, ic='aic')
# print(results.summary())


lag_order = results.k_ar
forecast_input = g1StockData.values[-(lag_order):]
forecast = results.forecast(y=forecast_input, steps=5)
forecast_df = pd.DataFrame(forecast, index=[f"Day {i}" for i in range(56, 61)], columns=g1StockData.columns)
g1StockData.index = [f"Day {i}" for i in range(1, 61)]
print("LAG:", lag_order, forecast_input)

for c in forecast_df.columns:
    g1StockData[c] = g1StockData[c].apply(lambda x: np.exp(x) if x != 0 else 0)
    forecast_df[c] = forecast_df[c].apply(lambda x: np.exp(x) if x != 0 else 0)

    # g1StockData[c] = lambdas[c].inverse_transform(g1StockData[[c]])
    # forecast_df[c] = lambdas[c].inverse_transform(forecast_df[[c]])

plt.figure(figsize=(12, 6))
plt.plot(g1StockData.iloc[:-5].index, g1StockData.iloc[:-5], label='Original')
plt.plot(forecast_df.index, forecast_df, label='Forecast')
# plt.legend()
plt.xticks(rotation=90)
plt.show()

correct = g1StockData.iloc[-1]
pred = forecast_df.iloc[-1]
prev = g1StockData.iloc[-2]
results = pd.DataFrame({"correct":correct, "pred":pred, "prev":prev})
print(correct)
print(pred)
print(results)

# Calculate MAE
mae = mean_absolute_error(results['correct'], results['pred'])
print('Mean Absolute Error:', mae)

# Calculate MSE
mse = mean_squared_error(results['correct'], results['pred'])
print('Mean Squared Error:', mse)

count = 0
for i in range(len(correct)):
    if correct.iloc[i] < prev.iloc[i] and pred.iloc[i] < prev.iloc[i]:
        count += 1
    if correct.iloc[i] > prev.iloc[i] and pred.iloc[i] > prev.iloc[i]:
        count += 1
print("Percent Directionally Correct:", count / len(correct), f"({count}/{len(correct)})")
