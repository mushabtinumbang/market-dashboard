import pandas as pd
import numpy as np
import sys
import os

from datetime import datetime
from loguru import logger
from src.utilities.scraper import dailyforex, economictimes
from src.utilities.config_ import scrape_data_path, log_path
import src.utilities.utils as utils

def run_scraper(
  date = ["01-01-2019"],      
  dailyfx = True,
  econtimes = True,
  suffix = 'new',
  dailyfx_out_feathername="dailyfx_.feather",
  econtimes_out_feathername="econtimes_.feather",
):
  """
  This is the main module to call the financial news scraper functions.
  Currently, there are only two sites available to scrape (might add more options in the future).
  1. Dailyfx (https://www.dailyfx.com/)
  2. Economic Times (https://economictimes.indiatimes.com/)
  """

  # Set formatted Date
  formatted_date = [details.strftime('%d-%m-%Y') for details in date]
  
  if dailyfx:
    # Starting logger for Daily Forex and show params
    logger.info("Starting to scrape Daily Forex ... \n")
    logger.info(
        "DailyFX Scraper Params- \n"
        + f" Date: {formatted_date} |\n"
        + f" File Output Suffix: {suffix} |\n"
    )

    # Run Scraper and save the result as df
    logger.info("Running Scraper ... \n")
    dailyforex_df = dailyforex(date)

    # Export Scrape result
    logger.info("Performing Writeout for DailyFX ... \n")

    utils.save(
      dailyforex_df,
      os.path.join(scrape_data_path, dailyfx_out_feathername)
    )

  if econtimes:
    # Starting logger for Econ Times and show params
    logger.info("Starting to scrape Economic Times ... \n")
    logger.info(
        "Economic Times Scraper Params- \n"
        + f" Date: {formatted_date} |\n"
        + f" File Output Suffix: {suffix} |\n"
    )

    # Run Scraper and save the result as df
    logger.info("Running Scraper ... \n")
    economictimes_df = economictimes(date)

    # Export Scrape result
    logger.info("Performing Writeout for Economic Times ... \n")

    utils.save(
      economictimes_df,
      os.path.join(scrape_data_path, econtimes_out_feathername)
    )


if __name__ == "__main__":
  run_scraper()