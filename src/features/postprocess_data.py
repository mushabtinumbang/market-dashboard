import pandas as pd
import numpy as np
import sys
import os
import joblib

from datetime import datetime
from loguru import logger
from src.utilities.scraper import dailyforex, economictimes
from src.utilities.config_ import predicted_data_path, combined_data_path
import src.utilities.utils as utils

def run_postprocess(
  dailyfx = True,
  econtimes = True,
  ftimes = True,
  dailyfx_pred_feathername= 'dailyfx_result_.feather',
  econtimes_pred_feathername= 'econtimes_result_.feather',
  financialtimes_pred_feathername = 'financialtimes_result_.feather',
  out_feathername= 'combined_data.feather'

):
    """
    After scraping the data and predict the sentiment classes, we will combine the temporary data to the previously combined data.
    The main goal of this step is to only have one data file that saves every news for every run. This is good to not lose any context.
    """
    # Create empty df
    temp_df = pd.DataFrame()

    # Run Post Processings
    logger.info("Starting to postprocess data ... \n")
    if dailyfx:
        # Read Predicted data
        logger.info("Combining DailyFX Data ... \n")
        dailyfx_pred = utils.load(os.path.join(predicted_data_path, dailyfx_pred_feathername))
        dailyfx_pred["source"] = "dailyfx"
        temp_df = pd.concat([temp_df, dailyfx_pred]).drop_duplicates().reset_index(drop=True)

    if econtimes:
        # Read Econtimes
        logger.info("Combining Economic Times Data ... \n")
        econtimes_pred = utils.load(os.path.join(predicted_data_path, econtimes_pred_feathername))
        econtimes_pred["source"] = "econtimes"
        temp_df = pd.concat([temp_df, econtimes_pred]).drop_duplicates().reset_index(drop=True)

    if ftimes:
        # Read Financial Times Data
        logger.info("Combining Financial Times Data ... \n")
        ftimes_pred = utils.load(os.path.join(predicted_data_path, financialtimes_pred_feathername))
        ftimes_pred["source"] = "financialtimes"
        temp_df = pd.concat([temp_df, ftimes_pred]).drop_duplicates().reset_index(drop=True)

    # check current combined file
    try:
        # read combined file, will produce error if not found.
        combined_file = utils.load(os.path.join(combined_data_path, out_feathername))

        # update data
        new_combined = pd.concat([combined_file, temp_df]).drop_duplicates().reset_index(drop=True)
    
        # writeout
        logger.info("Updating Combined File ... \n")
        utils.save(
            new_combined,
            os.path.join(combined_data_path, out_feathername)
        )
        
    except FileNotFoundError:
        logger.info("Combined File Currently Doesn't Exist, writing one ... \n")
        utils.save(
            temp_df,
            os.path.join(combined_data_path,out_feathername)
        )

    logger.info("Postprocessing steps complete! ... \n")


if __name__ == "__main__":
    run_postprocess()