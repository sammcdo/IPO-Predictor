from scraper import scrape
from timeseries import makedata


if __name__ == "__main__":
    data = scrape()

    symbols = list(data["Symbol"])
    offers = list(data["Offer Price"])

    stocks = makedata(symbols, offers)