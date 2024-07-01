import pandas as pd
import numpy as np
import sys
import os
import joblib

from datetime import datetime
from loguru import logger
from transformers import BertTokenizer, BertForSequenceClassification
from src.utilities.scraper import dailyforex, economictimes
from src.utilities.config_ import scrape_data_path, model_path, predicted_data_path, finbert_model_path
import src.utilities.utils as utils 

def run_forecast( 
  dailyfx = True,
  econtimes = True,
  ftimes = True,
  suffix = 'new',
  dailyfx_scrape_feathername= 'dailyfx_.feather',
  econtimes_scrape_feathername= 'econtimes_.feather',
  financialtimes_scrape_feathername = 'financialtimes_.feather',
  dailyfx_out_feathername= 'dailyfx_result_.feather',
  econtimes_out_feathername= 'econtimes_result_.feather',
  financialtimes_out_feathername = 'financialtimes_result_.feather'
):
    """
    This is the main module to predict previously scraped financial news..
    We will only be using FinBERT model since this is the best algorithm experimented for now. 
    """
    
    # Load the model
    logger.info("Loading FinBERT Model ... \n")
    finbert_model = BertForSequenceClassification.from_pretrained(finbert_model_path)

    # Load the tokenizer
    logger.info("Loading FinBERT Tokenizer ... \n")
    finbert_tokenizer = BertTokenizer.from_pretrained(finbert_model_path)

    # Run Prediction
    
    if dailyfx:
        # Starting logger for Daily Forex and show params
        logger.info("Starting to predict Daily Forex feather ... \n")
        logger.info(
            "DailyFX Forecasting Params- \n"
            + f" Input File Name: {dailyfx_scrape_feathername} |\n"
             + f" File Output Suffix: {suffix} |\n"
        )
        
        # load feather
        logger.info("Loading feather ... \n")
        dailyfx_df = utils.load(os.path.join(scrape_data_path, dailyfx_scrape_feathername))

        # run predict function
        logger.info(f"Starting Prediction for {dailyfx_df.shape[0]} news ... \n")
        dailyfx_result = utils.predict_with_finbert(
            df=dailyfx_df,
            loaded_model=finbert_model,
            loaded_tokenizer=finbert_tokenizer
        )

        # writing out..
        logger.info(f"Performing Writeout for Dailyfx Prediction feather ... \n")
        utils.save(
            dailyfx_result,
            os.path.join(predicted_data_path, dailyfx_out_feathername)
        )

    if econtimes:
        # Starting logger for Economic Times and show params
        logger.info("Starting to predict Economic Times feather ... \n")
        logger.info(
            "Economic Times Forecasting Params- \n"
            + f" Input File Name: {econtimes_scrape_feathername} |\n"
             + f" File Output Suffix: {suffix} |\n"
        )
        
        # load feather
        logger.info("Loading feather ... \n")
        econtimes_df = utils.load(os.path.join(scrape_data_path, econtimes_scrape_feathername))

        # run predict function
        logger.info(f"Starting Prediction for {econtimes_df.shape[0]} news ... \n")
        econtimes_result = utils.predict_with_finbert(
            df=econtimes_df,
            loaded_model=finbert_model,
            loaded_tokenizer=finbert_tokenizer
        )

        # writing out..
        logger.info(f"Performing Writeout for Economic Times Prediction feather ... \n")
        utils.save(
            econtimes_result,
            os.path.join(predicted_data_path, econtimes_out_feathername)
        )

    if ftimes:
        # Starting logger for Financial Times and show params
        logger.info("Starting to predict Financial Times feather ... \n")
        logger.info(
            "Financial Times Forecasting Params- \n"
            + f" Input File Name: {financialtimes_scrape_feathername} |\n"
             + f" File Output Suffix: {suffix} |\n"
        )

        # load feather
        logger.info("Loading feather ... \n")
        financialtimes_df = utils.load(os.path.join(scrape_data_path, financialtimes_scrape_feathername))

        # run predict function
        logger.info(f"Starting Prediction for {financialtimes_df.shape[0]} news ... \n")
        financialtimes_result = utils.predict_with_finbert(
            df=financialtimes_df,
            loaded_model=finbert_model,
            loaded_tokenizer=finbert_tokenizer
        )

        # writing out..
        logger.info(f"Performing Writeout for Financial Times Prediction feather ... \n")
        utils.save(
            financialtimes_result,
            os.path.join(predicted_data_path, financialtimes_out_feathername)
        )


if __name__ == "__main__":
    run_forecast()