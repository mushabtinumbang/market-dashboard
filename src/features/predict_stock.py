import pandas as pd
import os
import src.utilities.utils as utils

from loguru import logger
from src.utilities.config_ import summarized_data_path, model_path, predicted_data_path, distilbert_model_path
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer


def run_stock_prediction(
        df_summarized_out_feathername,
        df_predicted_out_feathername,
        stock
):
    """
    This is the main module to call the stock news sentiment prediction functions.
    Currently, there are only several stocks available to scrape (might add more options in the future).
    1. NVDA
    2. IBM
    3. AAPL
    4. INTL
    """
    # Load the model
    logger.info("Loading DistilBERT Model ... \n")
    distilbert_model = DistilBertForSequenceClassification.from_pretrained(distilbert_model_path)

    # Load the tokenizer
    logger.info("Loading DistilBERT Tokenizer ... \n")
    distilbert_tokenizer = DistilBertTokenizer.from_pretrained(distilbert_model_path)

    # Starting logger for sentiment prediction and show params
    logger.info("Starting to predict sentiment news feather ... \n")
    logger.info(
        "Sentiment Forecasting Params- \n"
        + f" Input File Name: {df_summarized_out_feathername} |\n"
            + f" Output File Name: {df_predicted_out_feathername} |\n"
    )
    
    # load feather
    logger.info("Loading feather ... \n")
    summarized_df = utils.load(os.path.join(summarized_data_path, df_summarized_out_feathername))

    # run predict function
    logger.info(f"Starting Prediction for {summarized_df.shape[0]} news ... \n")
    predicted_df = utils.predict_with_distilbert(
        df=summarized_df,
        loaded_model=distilbert_model,
        loaded_tokenizer=distilbert_tokenizer
    )

    # writing out..
    logger.info(f"Performing Writeout for stock sentiment prediction feather ... \n")
    utils.save(
        predicted_df,
        os.path.join(predicted_data_path, df_predicted_out_feathername)
    )


if __name__ == "__main__":
  run_stock_prediction()