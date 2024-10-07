import pandas as pd
import os

from loguru import logger
from src.utilities.config_ import scrape_data_path, summarized_data_path, model_path
import src.utilities.utils as utils

def run_stock_summarizer(
        df_scrape_out_feathername,
        df_summarized_out_feathername,
        stock
):
    """
    This is the main module to call the stock news summarizer functions.
    Currently, there are only several stocks available to scrape (might add more options in the future).
    1. NVDA
    2. IBM
    3. AAPL
    4. INTL
    """
    # Log
    logger.info("Starting to summarized scraped data ... \n")
    logger.info(
        "Investing.com Scraper Params- \n"
        + f" df_scrape_out_feathername: {df_scrape_out_feathername} |\n"
        + f" df_summarized_out_feathername: {df_summarized_out_feathername} |\n"
    )

    # Loading BART Tokenizer and Model
    logger.info("Loading BART Tokenizer and Model ... \n")
    tokenizer, model = utils.get_bart(model_path)

    # Load Scraped Data
    logger.info("Loading Scraped Data ... \n")
    df_data = utils.load(
       os.path.join(
          scrape_data_path, df_scrape_out_feathername
        )
    )

    # Run Summarizer with BART
    logger.info("Running Summarizer with BART... \n")
    df_data['summary'] = df_data['content'].apply(
                            lambda x: utils.summarize_with_bart(tokenizer, model, x) 
                                    if x != "None" else x)

    # Export Scrape result
    logger.info(f"Performing Writeout for summarized news content for {stock} ... \n")
    utils.save(
      df_data,
      os.path.join(summarized_data_path, df_summarized_out_feathername)
    )


if __name__ == "__main__":
  run_stock_summarizer()