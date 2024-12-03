import pandas as pd
import yfinance as yf


def getTimeseries(target="ROMA", offer=4.0, debug=False):
    # read data from yahoo finance
    data = yf.download(target, period='ytd', interval='1d')
    data.columns = data.columns.get_level_values('Price')

    # index is 30 days and the 60th day
    ind = pd.Index([f"Day {i}" for i in range(1,31)])
    ind = ind.append(pd.Index(["Day 60"]))

    cols = ["Open", "High", "Low", "Close"]

    output = pd.DataFrame(index=ind, columns=cols)

    # put desired data in output
    if len(data) > 60:
        print(len(output))
        # save all columns in the output data
        for i in range(len(cols)):
            output.iloc[0:30, i] = data.iloc[0:30][cols[i]].values
            output.iloc[30, i] = data.iloc[59][cols[i]]

            # convert output to percent of offer price
            output[cols[i]] = (output[cols[i]] - offer) / offer + 1
        if debug:
            print(output)
    
    return output

def makedata(symbols):
    data = {}
    for s in symbols:
        x = getTimeseries(target=s)
        if not x.isnull().values.any():
            data[s] = x
    
    combined = pd.concat(data, axis=1, keys=data.keys())
    
    print(combined)

    combined.to_csv("stocks.csv")

    return combined


if __name__ == "__main__":
    data = pd.read_csv("data.csv")
    symbols = list(data["Symbol"])

    makedata(symbols)