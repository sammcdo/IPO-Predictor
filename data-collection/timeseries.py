import pandas as pd
import yfinance as yf


def getTimeseries(target="ROMA", offer=4.0, debug=False):
    # read data from yahoo finance
    data = yf.download(target, period='ytd', interval='1d')
    data.columns = data.columns.get_level_values('Price')

    # index is 30 days and the 60th day
    ind = pd.Index([f"Day {i}" for i in range(1,56)])
    ind = ind.append(pd.Index(["Day 60"]))

    # save all the parts of the candle
    cols = ["Open", "High", "Low", "Close"]

    output = pd.DataFrame(index=ind, columns=cols)

    # put desired data in output
    if len(data) > 60:
        # save all columns in the output data
        for i in range(len(cols)):
            output.iloc[0:55, i] = data.iloc[0:55][cols[i]].values
            output.iloc[55, i] = data.iloc[59][cols[i]]

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