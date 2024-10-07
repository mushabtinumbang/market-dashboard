import pandas as pd
import os

from loguru import logger
from src.utilities.config_ import predicted_data_path, combined_data_path
import src.utilities.utils as utils


def run_postprocess_stock(
        out_feathername,
        df_predicted_out_feathername,
        stock
):
    """
    After scraping the data and predict the sentiment classes, we will combine the temporary data to the previously combined data.
    The main goal of this step is to only have one data file that saves every news for every run. This is good to not lose any context.
    """
    # Create empty df
    temp_df = pd.DataFrame()

    # Run Post Processings
    logger.info(f"Starting to postprocess data for {stock}... \n")

    # Read Predicted data
    logger.info("Combining Stock Data ... \n")
    stock_pred_df = utils.load(os.path.join(predicted_data_path, df_predicted_out_feathername))

    # Combine DataFrames
    temp_df = pd.concat([temp_df, stock_pred_df])

    # Ensure columns are of the correct type
    temp_df['date'] = temp_df['date'].astype(str)
    temp_df['title'] = temp_df['title'].astype(str)

    # Preprocess the 'content' column
    temp_df['content'] = temp_df['content'].apply(utils.preprocess_text)

    # Drop duplicates based on the 'title' column and reset index
    temp_df = temp_df.drop_duplicates(subset="title").reset_index(drop=True)

    # Check current combined file
    try:
        # Read combined file, will produce an error if not found
        combined_file = utils.load(os.path.join(combined_data_path, out_feathername))

        # Update data
        new_combined = pd.concat([combined_file, temp_df]).drop_duplicates(subset=["title", "category"]).reset_index(drop=True)

        # Write out the updated combined file
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
  run_postprocess_stock()