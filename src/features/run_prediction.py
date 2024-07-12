import pandas as pd
import numpy as np
import sys
import os
import joblib

from datetime import datetime
from loguru import logger
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from src.utilities.config_ import scrape_data_path, predicted_data_path, distilbert_model_path
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
    We will only be using DistilBERT model since this is the best algorithm experimented for now. 
    """
    
    # Load the model
    logger.info("Loading DistilBERT Model ... \n")
    distilbert_model = DistilBertForSequenceClassification.from_pretrained(distilbert_model_path)

    # Load the tokenizer
    logger.info("Loading DistilBERT Tokenizer ... \n")
    distilbert_tokenizer = DistilBertTokenizer.from_pretrained(distilbert_model_path)


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
        dailyfx_result = utils.predict_with_distilbert(
            df=dailyfx_df,
            loaded_model=distilbert_model,
            loaded_tokenizer=distilbert_tokenizer
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
        econtimes_result = utils.predict_with_distilbert(
            df=econtimes_df,
            loaded_model=distilbert_model,
            loaded_tokenizer=distilbert_tokenizer
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
        financialtimes_result = utils.predict_with_distilbert(
            df=financialtimes_df,
            loaded_model=distilbert_model,
            loaded_tokenizer=distilbert_tokenizer
        )

        # writing out..
        logger.info(f"Performing Writeout for Financial Times Prediction feather ... \n")
        utils.save(
            financialtimes_result,
            os.path.join(predicted_data_path, financialtimes_out_feathername)
        )


if __name__ == "__main__":
    run_forecast()