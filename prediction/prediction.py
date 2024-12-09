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

from prediction.prediction_util import *

import warnings

warnings.filterwarnings('ignore', category=InterpolationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

stocks = pd.read_csv("../data-collection/stocks.csv", header=[0,1], index_col=[0])
ipos = pd.read_csv("../analysis/clustered.csv")

# group based on clusters
group2 = ipos[ipos["Cluster"] == 0]
group1 = ipos[ipos["Industry_Cluster"] == 1]

close = stocks.xs('Close', axis=1, level=1)
close.index = stocks.index


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
    print("Test Statistic:", round(ts, 5), "\t", round(tsK, 5))
    print("P-Value:       ", round(p, 5), "\t", round(pK, 5))
    print("Lags:          ", lags, "\t", lagsK)
    print("DurbinWatson: %.5f" % db)
    others = [x for x in g1StockData.columns if x != c]
    gc = {}
    print()
    for x in others:
        gc[x] = min(granger_test(g1StockData[c], g1StockData[x]).values())
        print(x, gc[x])
    print(gc)
    if p > 0.05 and pK < 0.05:
        g1StockData.drop(columns=c, inplace=True)
    

print(g1StockData)

for c in g1StockData.columns:
    plt.plot(g1StockData.index, g1StockData[c].apply(lambda x: np.exp(x) if x != 0 else 0), label=c)

plt.legend()
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

plt.figure(figsize=(9, 4.8))
for i, c in enumerate(g1StockData.columns):
    plt.plot(g1StockData.iloc[:-5].index, g1StockData[c].iloc[:-5], label=c)
for i, c in enumerate(g1StockData.columns):
    plt.plot(forecast_df.index, forecast_df.iloc[:, i], color=plt.gca().lines[i].get_color())
# plt.plot(g1StockData.iloc[:-5].index, g1StockData.iloc[:-5], label='Original')
# plt.plot(forecast_df.index, forecast_df, label='Forecast')
plt.legend()
plt.xticks(rotation=90)
plt.show()

correct = g1StockData.iloc[-1]
pred = forecast_df.iloc[-1]
prev = g1StockData.iloc[-2]
error = abs(correct - pred)
errorPct = error / correct * 100
results = pd.DataFrame({"correct":correct, "pred":pred, "prev":prev, "errorPct":errorPct, "error":error})
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
errorPct = []
for i in range(len(correct)):
    if correct.iloc[i] < prev.iloc[i] and pred.iloc[i] < prev.iloc[i]:
        count += 1
    if correct.iloc[i] > prev.iloc[i] and pred.iloc[i] > prev.iloc[i]:
        count += 1
    errorPct.append(abs(correct.iloc[i] - pred.iloc[i]) / correct.iloc[i] * 100)
print("Percent Directionally Correct:", count / len(correct), f"({count}/{len(correct)})")
print("MAE of Percent Error:", sum(results["errorPct"]) / len(results))
print()
print(errorPct)
