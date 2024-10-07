import time
import pandas as pd
import click
import os
import concurrent.futures

from datetime import datetime, timedelta
from loguru import logger
from selenium import webdriver
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from src.utilities.config_ import scrape_data_path
from src.utilities.utils import format_dates, read_yaml


@click.command()
@click.option(
    "--date",
    "-e",
    required=False,
    type=str,
    help="Define 'latest', one specific date '24-01-2024', or date ranges '24-01-2024|26-01-2024'",
)
@click.option(
    "--suffix",
    "-s",
    required=False,
    type=str,
    default="",
    help="Suffix for the output names.",
)
def main_selenium(
    date,
    suffix
):
    # Log
    logger.info("Starting to scrape Investing.com ... \n")
    logger.info(
        "Investing.com Scraper Params- \n"
        + f" Pages: {date} |\n"
        + f" File Output Suffix: {suffix} |\n"
    )
    # Run Scraper with Selenium
    df_data = pd.DataFrame(run_investing_scrape(format_dates(date)))
    
    # Export df
    df_data.to_csv(os.path.join(scrape_data_path, f"investing_{suffix}.csv"))
    print(df_data)

def get_driver():
    # Setup Driver Options
    chrome_options = Options()
    chrome_options.page_load_strategy = 'none'
    driver = webdriver.Chrome(options=chrome_options)
    driver.set_window_size(700,2000)
    return driver

def run_investing_scrape(date):
    
    # Base URL and Final Data Structure
    BASE_URL = "https://www.investing.com/equities/nvidia-corp-news/"
    data, scrape_status, counter = {"title": [], "date": [], "url": [], "category": [], "content": []}, True, 0
    
    if len(date) == 2:
        date_start, date_end = date[0], (date[1] + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        date_start = date[0].replace(hour=0, minute=0, second=0, microsecond=0)
        date_end = (date_start + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)

    def scrape_article_content(url):
        """Helper function to scrape the content of a news article."""
        article_text = get_news_text(url)
        return article_text if article_text else "None"

    while scrape_status:
        # Log
        counter += 1
        logger.info(f"Scraping Page {counter} ...")

        driver = get_driver()
        current_url = BASE_URL + str(counter)
        driver.get(current_url)
        time.sleep(10)
        print(f"URL: {current_url}")
        
        try:
            # Wait for an element (like a news article) to load on the page
            soup = BeautifulSoup(driver.page_source, 'lxml')
            ul_element = soup.find('ul', {'data-test': 'news-list'})

            # Check if the ul_element is found
            if ul_element:
                # List to hold all URLs for parallel processing
                urls_to_scrape = []

                # Loop through each li with specific class
                for li in ul_element.find_all('div', class_='block w-full sm:flex-1'):
                    
                    # Check for 'a' tag with the 'article-title-link' class inside this li
                    a_tag = li.find('a', {'data-test': 'article-title-link'})
                    news_time = datetime.strptime(li.find('time', class_ = 'ml-2')['datetime'], '%Y-%m-%d %H:%M:%S')
                    category = "NVDA"
                    
                    try:
                        # if the date is earlier than date_start, stop scraping
                        if news_time < date_start:
                            scrape_status = False
                            break
                        
                        # if the date is within the range, scrape the data
                        if date_start <= news_time <= date_end:
                            try:
                                data["title"].append(a_tag.text.strip())
                                data["url"].append(a_tag['href'])
                                data["category"].append(category) 
                                data["date"].append(news_time)

                                # Collect URLs for parallel scraping
                                urls_to_scrape.append(a_tag['href'])

                                # If the <a> tag is found, print its text and href
                                print(f"Success: Title: {a_tag.text.strip()}")
                                print(f"URL: {a_tag['href']}")
                                print(f"Datetime: {news_time}\n")

                            except Exception as e:
                                print(f"Failed to extract text or URL from li: {li}\nError: {e}")
                                
                    except Exception as e2:
                        print(f"No article link found in li: {li}\n")

                # Use ThreadPoolExecutor to scrape all URLs in parallel
                with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                    # Submit all URLs for parallel scraping and collect results
                    article_contents = list(executor.map(scrape_article_content, urls_to_scrape))

                # Append the content for each article to the data dictionary
                data["content"].extend(article_contents)

        except Exception as e3:
            logger.error(f"Error loading page: {e3}")
        finally:
            driver.quit()

    return data


def get_news_text(url):
    """Function to open each news article URL and scrape its text."""
    driver = get_driver()
    driver.get(url)
    time.sleep(10)

    # Scraping the article text..
    print(f"Scraping text for URL: {url}")

    try:
        # Find the div with id='article' and class='article_container'
        soup = BeautifulSoup(driver.page_source, 'lxml')
        article_div = soup.find('div', {'id': 'article', 'class': 'article_container'})

        # Check if the div was found
        if article_div:
            # Get all <p> tags inside the div
            paragraphs = article_div.find_all('p')
            
            # Extract and join the text from each <p> tag
            article_text = "\n".join(p.get_text() for p in paragraphs)
            
            # Quit the driver before returning
            driver.quit()
            
            return article_text
        
        else:
            raise Exception("Article div not found.")
    
    except Exception as error:
        print(f"Fail to extract text for URL {url}: {error}")
        driver.quit()  # Ensure the driver is quit even in case of errors
        return None  # Return None if scraping fails


if __name__ == "__main__":
    main_selenium()