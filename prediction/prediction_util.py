import pandas as pd
import statsmodels.api as sm
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss, InterpolationWarning, grangercausalitytests
from statsmodels.stats.stattools import durbin_watson

import warnings

warnings.filterwarnings('ignore', category=InterpolationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

def adf_test(series):
    """ returns test statistic and p-value and lags """
    r = adfuller(series, regression='ct')
    return r[0], r[1], r[2]

def kpss_test(series):
    """ returns test statistic and p-value and lags """
    r = kpss(series, regression='ct')
    return r[0], r[1], r[2]

def durbinWatson_test(series):
    """ returns the p-value"""
    df=pd.DataFrame()
    df["1"] = series
    df["2"] = df["1"].shift(1)
    df.dropna(inplace=True)
    model = sm.OLS(df["1"], df["2"]).fit()
    residuals = model.resid
    r = durbin_watson(residuals)
    return r

def granger_test(series1, series2):
    """ returns the p-value"""
    data = pd.DataFrame({'series1': series1, 'series2': series2})
    test_result = grangercausalitytests(data[['series2', 'series1']], 5, verbose=False)
    p_values = {lag: float(test_result[lag][0]['ssr_ftest'][1]) for lag in range(1, 5 + 1)}
    return p_values

def getGroupClosingPrices(group, close):
    # separate just the closing data for each stock
    out = close.copy(deep=True)
    g1Symbols = [str(i) for i in group["symbol"]]
    group1Stocks = [i for i in out.columns if str(i) in g1Symbols]
    out = out[group1Stocks]
    return out