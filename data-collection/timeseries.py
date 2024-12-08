import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler


def getTimeseries(target="ROMA", debug=False):
    """
    Download a time series from Yahoo Finance
    Params:
    target - The symbol to download
    debug - Set to true for more detailed print statements
    Return:
    A dataframe with the OHLC and volume data for the given stock
    """
    # read data from yahoo finance
    data = yf.download(target, period='ytd', interval='1d')
    data.columns = data.columns.get_level_values('Price')
    # index is 60 days
    ind = pd.Index([f"Day {i}" for i in range(1,61)])

    # save all the parts of the candle
    cols = ["Open", "High", "Low", "Close", "Volume"]

    output = pd.DataFrame(index=ind, columns=cols)

    # put desired data in output
    if len(data) > 60:
        # save all columns in the output data
        for i in range(len(cols)):
            output.iloc[0:60, i] = data.iloc[0:60][cols[i]].values
        if debug:
            print(output)
    
    return output


def makedata(symbols):
    """
    Given a list of symbols, get their stock data and 
    combine into one master dataframe
    Params:
    symbols - A list of tickers to download data for
    Return:
    A dataframe of stock data
    """
    ipo = pd.read_csv("data.csv")
    data = {}
    for s in range(len(symbols)):
        x = getTimeseries(target=symbols[s])
        if not x.isnull().values.any():
            data[symbols[s]] = x
    
    combined = pd.concat(data, axis=1, keys=data.keys())
    
    print(combined)

    combined.to_csv("stocks.csv")

    return combined


### Tests
if __name__ == "__main__":
    data = pd.read_csv("data.csv")
    symbols = list(data["Symbol"])

    makedata(symbols)