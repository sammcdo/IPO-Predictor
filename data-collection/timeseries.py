import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler


def getTimeseries(target="ROMA", offer=4.0, debug=False):
    # read data from yahoo finance
    data = yf.download(target, period='ytd', interval='1d')
    data.columns = data.columns.get_level_values('Price')

    # index is 60 days
    ind = pd.Index([f"Day {i}" for i in range(1,61)])

    # save all the parts of the candle
    cols = ["Open", "High", "Low", "Close"]

    output = pd.DataFrame(index=ind, columns=cols)

    # put desired data in output
    if len(data) > 60:
        # save all columns in the output data
        for i in range(len(cols)):
            output.iloc[0:60, i] = data.iloc[0:60][cols[i]].values

            # do minmax scaling on price data
            # s = MinMaxScaler()
            # output[cols[i]] = s.fit_transform(output[[cols[i]]])

            # convert output to percent of offer price
            # output[cols[i]] = (output[cols[i]] - offer) / offer + 1
        if debug:
            print(output)
    
    return output

def makedata(symbols, openings):
    ipo = pd.read_csv("data.csv")
    data = {}
    for s in range(len(symbols)):
        x = getTimeseries(target=symbols[s], offer=openings[s])
        if not x.isnull().values.any():
            data[symbols[s]] = x
    
    combined = pd.concat(data, axis=1, keys=data.keys())
    
    print(combined)

    combined.to_csv("stocks.csv")

    return combined


if __name__ == "__main__":
    data = pd.read_csv("data.csv")
    symbols = list(data["Symbol"])
    offers = list(data["Offer Price"])

    makedata(symbols, offers)