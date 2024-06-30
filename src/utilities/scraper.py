import requests
import json
import pandas as pd
import re

from datetime import datetime
from bs4 import BeautifulSoup
from loguru import logger
from src.utilities.utils import format_dates, date_to_integer

def dailyforex(
        date
):
    base_url, data = "https://www.dailyfx.com/archive/" , {"title" : [], "date" : [], "url" : [], "category" : []}
    
    month_start, year_start = date[0].month, date[0].year

    if len(date) == 1:
        month_end, year_end = month_start + 1, year_start + 1

    else:
        month_end = month_start + 1 if (date[1].month == month_start) else date[1].month + 1
        year_end = year_start + 1 if (date[1].year == year_start) else date[1].year + 1

    # Loop over the years from 2019 to 2023
    for year in range(year_start, year_end):
        # Loop over the months from January to November
        for month in range(month_start, month_end):
            # Format the month as '01', '02', ..., '09', '10', '11'
            formatted_month = f"{month:02d}"
            
            # Create the complete URL
            url = f"{base_url}{year}/{formatted_month}"
            html_text = requests.get(url).text
            soup = BeautifulSoup(html_text, 'lxml')
            titles = soup.find_all('section', class_='my-6')
            
            for j in titles:
                urls = j.find_all('a')
                date = datetime.strptime(j.find('h2', class_ = 'text-black dfx-h-3').text, "%d %B, %Y (%A)").strftime("%Y-%m-%d")
                news = j.find_all("span", class_ = 'dfx-articleListItem__title')
                headlines, category = [], "forex"

                for k in range(len(news)):
                    data["date"].append(date)
                    data["url"].append(str(urls[k].get('href')))
                    data["category"].append(category)
                    data["title"].append(news[k].text)
                    headlines.append(news[k].text)

    # Export to DataFrame
    final_df = pd.DataFrame(data)

    return final_df


def economictimes(
        date
):
    # Parameters
    base_url, data = "https://www.wsj.com/news/archive/" , {"title" : [], "date" : [], "url" : [], "category" : []}
    HEADERS = {
    'User-Agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
    }

    # Get Integers from Date
    date_integers = date_to_integer(date)
    start = date_integers[0]
    
    # Handle Errors
    try:
        end = date_integers[1]
    
    except IndexError:
        end = start

    for index in range(start, end + 1):
        url = f'https://economictimes.indiatimes.com/archivelist/starttime-{index}.cms'
        response = requests.get(url, headers=HEADERS).text
        soup = BeautifulSoup(response, 'lxml')
        cols = soup.find_all('ul', class_='content')
        date_obj = soup.find('td', class_ = 'contentbox5').find_all('b')[1].text
        date = datetime.strptime(date_obj, "%d %b, %Y").strftime("%Y-%m-%d")

        for i in range(2):
            for j in cols[i]:
                # Get title and URL
                urlhead  = 'https://economictimes.indiatimes.com/'
                urlbody = str(j.find('a').get('href'))
                itemstitle = j.find('a').text
                pattern = r"\/([^\/]+)\/([^\/]+)\/([^\/]+)\/"

                try:
                    matches = re.search(pattern, urlbody).groups()
                    
                    for x in matches:
                        if x in ["banking", "economy", "market", "forex"]:
                            data["title"].append(itemstitle)
                            data["url"].append(urlhead + urlbody)
                            data["category"].append(x) 
                            data["date"].append(date)
                            break

                        else:
                            continue
                except:
                    continue

    # Export to DataFrame
    final_df = pd.DataFrame(data)

    return final_df