from statsmodels.tools.sm_exceptions import ValueWarning

from prediction.prediction_util import *
from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

import warnings
warnings.filterwarnings('ignore', category=ValueWarning)

stocks = pd.read_csv("data-collection/stocks.csv", header=[0,1], index_col=[0])
ipos = pd.read_csv("analysis/clustered.csv")

monthLabels = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

for INUM, ILABEL in enumerate(monthLabels):
    if INUM != 0:
        print("\n\n")
    print("IPO MONTH:", ILABEL)

    group1 = ipos[ipos["Month_Cluster"] == INUM]

    close = stocks.xs('Close', axis=1, level=1)
    close.index = stocks.index


    g1StockData = getGroupClosingPrices(group1, close)

    for c in g1StockData.columns:

        g1StockData[c] = g1StockData[c].apply(lambda x: np.log(x) if x != 0 else 0)

        ts, p, lags = adf_test(g1StockData[c].dropna())
        tsK, pK, lagsK = kpss_test(g1StockData[c].dropna())
        db = durbinWatson_test(g1StockData[c].dropna())
        print(c)
        print(f"Test Statistic:  ADF: {round(ts, 5)}\t KPSS: {round(tsK, 5)}")
        print(f"P-Value:         ADF: {round(p, 5)}\t KPSS: {round(pK, 5)}")
        print("DurbinWatson: %.5f" % db)
        others = [x for x in g1StockData.columns if x != c]
        gc = {}
        print("Granger Causality Tests")
        for x in others:
            gc[x] = min(granger_test(g1StockData[c], g1StockData[x]).values())
            print(x, gc[x])
        if p > 0.05 and pK < 0.05:
            g1StockData.drop(columns=c, inplace=True)

    if len(g1StockData.columns) < 2:
        print("Insufficient number of symbols after preprocessing")
        continue
    g1StockData = g1StockData.iloc[:, 0:15]
    date_labels = pd.date_range(start='2024-01-01', periods=60, freq='D')
    g1StockData["date"] = date_labels
    g1StockData.set_index("date", inplace=True)

    model = VAR(g1StockData.iloc[:-5])
    results = model.fit(maxlags=1, ic='aic')

    lag_order = results.k_ar
    forecast_input = g1StockData.values[-(lag_order):]
    forecast = results.forecast(y=forecast_input, steps=5)
    forecast_df = pd.DataFrame(forecast, index=[f"Day {i}" for i in range(56, 61)], columns=g1StockData.columns)
    g1StockData.index = [f"Day {i}" for i in range(1, 61)]

    for c in forecast_df.columns:
        g1StockData[c] = g1StockData[c].apply(lambda x: np.exp(x) if x != 0 else 0)
        forecast_df[c] = forecast_df[c].apply(lambda x: np.exp(x) if x != 0 else 0)

    plt.figure(figsize=(9, 4.8))
    for i, c in enumerate(g1StockData.columns):
        plt.plot(g1StockData.iloc[:-5].index, g1StockData[c].iloc[:-5], label=c)
    for i, c in enumerate(g1StockData.columns):
        plt.plot(forecast_df.index, list(forecast_df.iloc[:, i]), color=plt.gca().lines[i].get_color(), label=None)

    plt.legend(loc='upper left')
    plt.xticks(rotation=90)
    plt.show()
    g1StockData.replace([0], 0.001, inplace=True)

    correct = g1StockData.iloc[-1]
    pred = forecast_df.iloc[-1]
    prev = g1StockData.iloc[-2]
    error = abs(correct - pred)
    errorPct = error / correct * 100
    results = pd.DataFrame({"correct":correct, "pred":pred, "prev":prev, "errorPct":errorPct, "error":error})
    results.replace([np.inf, -np.inf], 100, inplace=True)
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