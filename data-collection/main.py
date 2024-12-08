"""
Sam McDowell
12/09/2024

Main script to scrape all data for this project
"""


from scraper import scrape
from timeseries import makedata


if __name__ == "__main__":
    data = scrape()

    symbols = list(data["Symbol"])

    stocks = makedata(symbols)