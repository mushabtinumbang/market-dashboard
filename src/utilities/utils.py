import pandas as pd
import numpy as np
import pickle
import feather
import gzip
import yaml
import re
import os 
import torch
import concurrent.futures

from datetime import datetime
from jinja2 import Template
from datetime import datetime, timedelta
from loguru import logger
from selenium import webdriver
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from src.utilities.config_ import *
from transformers import BartForConditionalGeneration, BartTokenizer
import concurrent.futures
from bs4 import BeautifulSoup

def format_dates(date_input):
    """
    This function takes a date input which could be 'latest', a single date,
    or a date range, and returns a list of dates in the datetime format.
    """
    today_date = datetime.today()

    if date_input.lower() == "latest":
        return [today_date]
    
    date_parts = date_input.split("|")
    
    if len(date_parts) == 1:
        return [datetime.strptime(date_parts[0], '%d-%m-%Y')]
    
    elif len(date_parts) == 2:
        return [
            datetime.strptime(date_parts[0], '%d-%m-%Y'),
            datetime.strptime(date_parts[1], '%d-%m-%Y')
        ]
    
    else:
        raise ValueError("Invalid date input format")
    
def date_to_integer(dates):
    """
    This function converts datetime objects to integer dates based on a reference date.
    Reference date: 43466 represents 01 January 2019.
    """
    reference_date = datetime(2019, 1, 1)
    reference_integer = 43466
    
    integer_dates = [(reference_integer + (date - reference_date).days) for date in dates]
    return integer_dates

def dates_to_string(dates_list):
    """
    This function takes a list of date strings and concatenates them into a single string.
    If there is only one date, it returns the date string.
    If there are multiple dates, it joins them with an underscore.
    """
    if len(dates_list) == 1:
        return dates_list[0]
    else:
        return '_'.join(dates_list)
    
def predict(
        df,
        loaded_svm_model,
        loaded_vectorizer
):
    # New text data to predict
    titles = list(df['title'])

    # Preprocess the new text data
    new_texts_tfidf = loaded_vectorizer.transform(titles)

    # Predict the labels for the new text data
    predictions = loaded_svm_model.predict(new_texts_tfidf)
    df["label"] = predictions

    return df

def save(data, filename):
    folders = os.path.dirname(filename)
    if folders:
        os.makedirs(folders, exist_ok=True)

    fl = filename.lower()
    if fl.endswith(".gz"):
        if fl.endswith(".feather.gz") or fl.endswith(".fthr.gz"):
            # Since feather doesn't support writing to the file handle, we
            # can't easily point it to gzip.
            raise NotImplementedError(
                "Saving to compressed .feather not currently supported."
            )
        else:
            fp = gzip.open(filename, "wb")
            pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        if fl.endswith(".feather") or fl.endswith(".fthr"):
            if str(type(data)) != "<class 'pandas.core.frame.DataFrame'>":
                raise TypeError(
                    ".feather format can only be used to save pandas "
                    "DataFrames"
                )
            feather.write_dataframe(data, filename)
        else:
            fp = open(filename, "wb")
            pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)


def load(filename):
    """
    Loads data saved with save() (or just normally saved with pickle).
    Autodetects gzip if filename ends in '.gz'
    Also reads feather files denoted .feather or .fthr.

    Parameters
    ----------
    filename -- String with the relative filename of the pickle/feather
    to load.
    """
    fl = filename.lower()
    if fl.endswith(".gz"):
        if fl.endswith(".feather.gz") or fl.endswith(".fthr.gz"):
            raise NotImplementedError("Compressed feather is not supported.")
        else:
            fp = gzip.open(filename, "rb")
            return pickle.load(fp)
    else:
        if fl.endswith(".feather") or fl.endswith(".fthr"):
            import feather

            return feather.read_dataframe(filename)
        else:
            fp = open(filename, "rb")
            return pickle.load(fp)


def read_yaml(filename, render=False, **kwargs):
    """
    Read yaml configuation and returns the dict

    Parameters
    ----------
    filename: string
        Path including yaml file name
    render: Boolean, default = False
        Template rendering
    **kwargs:
        Template render args to be passed
    """
    if render:
        yaml_text = Template(open(filename, "r").read())
        yaml_text = yaml_text.render(**kwargs)
        config = yaml.safe_load(yaml_text)
    else:
        with open(filename) as f:
            config = yaml.safe_load(f)

    return config


def predict_with_distilbert(
        df,
        loaded_model,
        loaded_tokenizer,
        batch_size=32
):
    # New text data to predict
    titles = list(df['title'])

    # Move model to the appropriate device (e.g., CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded_model.to(device)

    # Initialize an empty list to store predictions
    all_predictions = []

    # Process data in batches
    for i in range(0, len(titles), batch_size):
        batch_titles = titles[i:i + batch_size]
        
        # Tokenize the new text
        inputs = loaded_tokenizer(batch_titles, padding='max_length', truncation=True, max_length=64, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        
        # Make predictions
        loaded_model.eval()
        with torch.no_grad():
            outputs = loaded_model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
        
        # Convert predictions to labels
        int_to_label = {0: 'positive', 1: 'neutral', 2: 'negative'}
        predicted_labels = [int_to_label[pred.item()] for pred in predictions]
        all_predictions.extend(predicted_labels)

    # Add the predictions to the dataframe
    df["label"] = all_predictions
    
    return df


def get_driver():
    # Setup Driver Options
    chrome_options = Options()
    chrome_options.page_load_strategy = 'none'
    driver = webdriver.Chrome(options=chrome_options)
    driver.set_window_size(700,700)
    return driver


def run_investing_scrape(
        stock,
        date,
        base_url
    ):
    
    # Final Data Structure
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
        if counter > 100:
            scrape_status = False
            break
        else:
            counter += 1

        logger.info(f"Scraping Page {counter} ...")

        driver = get_driver()
        current_url = base_url + str(counter)
        driver.get(current_url)
        print(f"URL: {current_url}")
        
        try:
            # Explicitly wait for the 'ul' element containing news articles to load
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.XPATH, "//ul[@data-test='news-list']"))
            )

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
                    news_time = datetime.strptime(li.find('time', class_='ml-2')['datetime'], '%Y-%m-%d %H:%M:%S')
                    category = stock
                    
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
                with concurrent.futures.ThreadPoolExecutor(
                                    max_workers= 10 # Set based on your CPU and RAM, max value: 10.
                    ) as executor:
                    # Submit all URLs for parallel scraping and collect results
                    article_contents = list(
                        executor.map(
                            scrape_article_content, 
                            urls_to_scrape)
                    )

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

    try:
        # Wait for the article container to load (with a maximum wait time of 20 seconds)
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.ID, "article"))
        )
        
        # Scraping the article text..
        print(f"Scraping text for URL: {url}")

        # Find the div with id='article' and class='article_container'
        soup = BeautifulSoup(driver.page_source, 'lxml')
        # Find the div containing the article content
        article_div = soup.find('div', {'id': 'article', 'class': 'article_container'})

        # Check if the article div was found
        if article_div:
            # Get all <p> tags inside the div
            paragraphs = article_div.find_all('p')
            
            # Loop through each p tags
            article_text = []
            for p in paragraphs:
                # Skip <p> tags that contain an <img> tag
                if p.find('img'):
                    continue
                
                # Find the parent div with the attribute data-test="contextual-subscription-hook-text"
                parent_div = p.find_parent('div', {'data-test': 'contextual-subscription-hook'})
                
                # If the <p> is inside such a div, skip it
                if parent_div:  
                    continue
                
                # Append the paragraph text to the article_text list
                article_text.append(p.get_text())

            # Join all the paragraph texts
            article_text = "\n".join(article_text)
            
            # Quit the driver before returning
            driver.quit()
            
            return article_text
        
        else:
            raise Exception("Article div not found.")
    
    except Exception as error:
        print(f"Failed to extract text for URL {url} ...")
        driver.quit()  # Ensure the driver is quit even in case of errors
        return None  # Return None if scraping fails

def summarize_with_bart(
        tokenizer,
        model,
        text
):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)

    # Generate summary
    summary_ids = model.generate(inputs["input_ids"], max_length=100, min_length=25, length_penalty=2.0, num_beams=4, early_stopping=True)

    # Decode the summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary

def get_bart(model_path):
    # Load the tokenizer
    tokenizer = BartTokenizer.from_pretrained(os.path.join(model_path, "bart-large-cnn"))
    # Load the model
    model = BartForConditionalGeneration.from_pretrained(os.path.join(model_path, "bart-large-cnn"))

    return tokenizer, model

# Function to preprocess text
def preprocess_text(text):
    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with a single space
    text = text.strip()  # Remove leading and trailing whitespace
    return text