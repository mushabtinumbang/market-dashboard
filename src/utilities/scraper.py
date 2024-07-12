import requests
import json
import pandas as pd
import re

from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from loguru import logger
from src.utilities.utils import format_dates, date_to_integer

def dailyforex(
        date
):
    base_url = "https://www.dailyfx.com/archive/"
    data = {"title" : [], "date" : [], "url" : [], "category" : []}
    
    month_start, year_start = date[0].month, date[0].year

    if len(date) == 1:
        month_end, year_end = month_start + 1, year_start + 1

    else:
        month_end = month_start + 1 if (
            date[1].month == month_start) else date[1].month + 1
        
        year_end = year_start + 1 if (
            date[1].year == year_start) else date[1].year + 1

    # Loop over the start and end date
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
                date = datetime.strptime(j.find('h2', 
                                        class_ = 'text-black dfx-h-3').
                                        text, "%d %B, %Y (%A)").strftime("%d-%m-%Y")
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


def economictimes(date):
    # Parameters
    data = {"title" : [], "date" : [], "url" : [], "category" : []}

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
        response = requests.get(url).text
        soup = BeautifulSoup(response, 'lxml')
        cols = soup.find_all('ul', class_='content')
        date_obj = soup.find('td', class_ = 'contentbox5').find_all('b')[1].text
        date = datetime.strptime(date_obj, "%d %b, %Y").strftime("%d-%m-%Y")

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

def financialtimes(date):
    base_url= "https://www.ft.com/currencies?page=" 
    data, i, scrape = {"title" : [], "date" : [], "url" : [], "category" : []}, 1, True
    
    if len(date) == 2:
        date_start, date_end = date[0], (date[1] + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        date_start = date[0].replace(hour=0, minute=0, second=0, microsecond=0)
        date_end = (date_start + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)

    while scrape:
        url = f"{base_url}{i}"
        html_text = requests.get(url).text
        soup = BeautifulSoup(html_text, 'lxml')
        all_titles = soup.find_all('li', class_='o-teaser-collection__item o-grid-row')

        for title in all_titles:
            try:
                # find data
                date_text = title.find('time', class_='o-date').get_text()
                date = datetime.strptime(date_text, '%A, %d %B, %Y')

                # if the date is earlier than date_start, stop scraping
                if date < date_start:
                    scrape = False
                    break
                
                # if the date is within the range, scrape the data
                if date_start <= date <= date_end:
                    category = title.find('a', class_='o-teaser__tag').get_text()
                    title_div = title.find('a', class_='js-teaser-heading-link')
                    title_url = f"ft.com{title_div.get('href')}"
                    header = title_div.get_text()

                    # append data
                    data["title"].append(header)
                    data["date"].append(date.strftime('%d-%m-%Y'))
                    data["url"].append(title_url)
                    data["category"].append(category)
                
                # if the date is later than date_end, continue to next title
                else:
                    continue

            except AttributeError as e:
                pass

        i += 1

    final_df = pd.DataFrame(data)
    
    return final_df