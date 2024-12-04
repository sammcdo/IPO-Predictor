"""
Sam McDowell
11/25/2024

Scraping www.iposcoop.com/current-year-pricings/
for IPO data and saving it to a CSV.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.preprocessing import StandardScaler

import subpage

def scrape():
    # URL of the webpage to scrape
    url = 'https://www.iposcoop.com/current-year-pricings/'

    # Send a GET request to fetch the webpage content
    response = requests.get(url, allow_redirects=True)
    response.raise_for_status()  # Check for request errors

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the table by its class
    table = soup.find('table', class_='ipolist')

    # Initialize lists to hold table data
    rows = {

    }

    cols = []

    # Find all the headers
    for row in table.find_all('tr'):
        cells = row.find_all(['th']) 
        for cell in cells:
            rows[cell.get_text(strip=True)] = []
            cols.append(cell.get_text(strip=True))

    # Find all the data
    for row in table.find_all('tr'): 
        cells = row.find_all(['td'])
        for c in range(len(cells)): # Check if the cell contains a link 
            cell = cells[c]
            link = cell.find('a', href=True)
            if link:
                col = cols[c] + "_link"
                if col not in rows:
                    rows[col] = []
                rows[col].append(link["href"].strip())
            rows[cols[c]].append(cell.get_text(strip=True))

    # Create a DataFrame from the extracted data
    df = pd.DataFrame(rows)

    for i, r in df.iterrows():
        url = r["Company_link"]
        results = subpage.getFromSubpage(url)
        for k,v in results.items():
            df.at[i,k] = v
        print(results)

    # drop unneeded columns
    df.drop(columns=[
        "Return",
        "SCOOP Rating", 
        "Current Price", 
        "Company_link",
        "Symbol_link",
        "Industry_link",
        "SCOOP Rating_link",
        "1st Day Close",
        "Est. $ Volume",      # not known before stock starts trading
        ], inplace=True)

    df["Offer Price"] = df["Offer Price"].apply(lambda x: float(x.replace("$","")))

    # Save the DataFrame to a CSV file
    df.to_csv('data_raw.csv', index=False)

    # scale the data
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        scaler = StandardScaler()
        df[col] = scaler.fit_transform(df[[col]])

    # save the scaled data
    df.to_csv('data.csv', index=False)

    # Print the DataFrame to verify
    print(df)

    return df


if __name__ == "__main__":
    scrape()