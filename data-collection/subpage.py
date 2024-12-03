import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

def getFromSubpage(url):
    """
    Using this URL scrape for IPO Financials
    Params: URL like https://www.iposcoop.com/ipo/rubrik-inc/
    Return: Dict of IPO Financial Data
    """
    response = requests.get(url)
    response.raise_for_status()  # Check for request errors

    soup = BeautifulSoup(response.content, 'html.parser')

    table = soup.find('table', class_='ipo-table')
    data = {
        "Market Cap": None,
        "Revenues": None,
        "Net Income": None,
        "Shares (millions)": None,
        "Est. $ Volume": None
    }

    # Loop through the rows of the table to extract data
    for row in table.find_all('tr'):
        cells = row.find_all('td')
        if len(cells) == 2:
            field = cells[0].get_text(strip=True).replace(':', '')
            value = cells[1].get_text(strip=True)
            try:
                value = float(re.search(r'\d+\.\d+', value).group())
            except:
                value = 0
            if field in data:
                data[field] = value

    return data


if __name__ == "__main__":
    url = "https://www.iposcoop.com/ipo/rubrik-inc/"
    data = getFromSubpage(url)

    print(data)
