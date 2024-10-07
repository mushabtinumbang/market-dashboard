import pandas as pd
import os

from loguru import logger
from src.utilities.config_ import scrape_data_path
import src.utilities.utils as utils


def run_scraper_stock(
        stock,
        date,
        df_out_feathername,
        suffix,
        url
):
    """
    This is the main module to call the stock news scraper functions.
    Currently, there are only several stocks available to scrape (might add more options in the future).
    1. NVDA
    2. IBM
    3. AAPL
    4. INTL
    """
    # Log
    logger.info("Starting to scrape Investing.com ... \n")
    logger.info(
        "Investing.com Scraper Params- \n"
        + f" Date: {date} |\n"
        + f" Stock: {stock} |\n"
        + f" File Output Suffix: {suffix} |\n"
    )

    # Run Scraper
    logger.info("Running Scraper ... \n")
    df_data = pd.DataFrame(
        utils.run_investing_scrape(
            stock,
            utils.format_dates(date),
            url
        )
    )

    # Export Scrape result
    logger.info(f"Performing Writeout for {stock} with {df_data.shape[0]} rows ... \n")
    utils.save(
      df_data,
      os.path.join(scrape_data_path, df_out_feathername)
    )


if __name__ == "__main__":
  run_scraper_stock()