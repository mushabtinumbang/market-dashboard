import pandas as pd
import numpy as np
import os 
import sys
import click

from loguru import logger
from datetime import datetime
from src.features.run_scraper import run_scraper
from src.features.run_prediction import run_forecast
from src.features.postprocess_data import run_postprocess
from src.utilities.config_ import log_path, ConfigManager, config_path
from src.utilities.utils import format_dates, read_yaml

@click.command()
@click.option(
    "--date",
    "-e",
    required=True,
    type=str,
    help="Define 'latest', one specific date '24-01-2024', or date ranges '24-01-2024|26-01-2024'",
)
@click.option(
    "--dailyfx",
    required=True,
    type=str,
    help="Define whether the user wants to run scraper for Dailyfx or not. 'y' for yes and 'n' for no.",
)
@click.option(
    "--econtimes",
    required=True,
    type=str,
    help="Define whether the user wants to run scraper for Economic Times or not. 'y' for yes and 'n' for no.",
)
@click.option(
    "--suffix",
    "-s",
    required=False,
    type=str,
    default="",
    help="Suffix for the output names.",
)
@click.option(
    "--pipeline",
    "-p",
    type=click.Choice(
        ["y", "n"],
        case_sensitive=False,
    ),
    default=["n", "n", "n"],
    show_default=True,
    multiple=True,
    help="example: -p y -p y -p y\n"
    " 1st p = scrape financial news;"
    " 2nd p = predict sentiments;"
    " 3nd p = prepare data for streamlit;"
)
def main_predict_sentiments(
    date,
    dailyfx,
    econtimes,
    suffix,
    pipeline,
):
    container_date = datetime.now().strftime("%Y-%m-%d")
    logger.remove()
    logger.add(
        os.path.join(log_path, "run_predict_sentiment" + container_date + ".log"),
        format="<green>{time}</green> | <yellow>{name}</yellow> | {level} |"
        " <cyan>{message}</cyan>"
    )
    logger.add(
        sys.stderr,
        colorize=True,
        format="<green>{time}</green> | <yellow>{name}</yellow> | {level} |"
        " <cyan>{message}</cyan>"
    )

    # load some config
    params = read_yaml(
        os.path.join(config_path, "main_config.yaml"), render=True, suffix=suffix
    )

    if pipeline[0] == "y":
        # define parameters
        scraper_params = params["run_scraper_pipeline_params"]
        scraper_params["date"] = format_dates(date)
        scraper_params["dailyfx"] = True if dailyfx == 'y' else False
        scraper_params["econtimes"] = True if dailyfx == 'y' else False
        scraper_params["suffix"] = suffix

        # run scraper
        run_scraper(**scraper_params)

    if pipeline[1] == "y":
        # define parameters
        predict_params = params["run_prediction_pipeline_params"]
        predict_params["dailyfx"] = True if dailyfx == 'y' else False
        predict_params["econtimes"] = True if dailyfx == 'y' else False
        predict_params["suffix"] = suffix

        # run prediction
        run_forecast(**predict_params)

    if pipeline[2] == "y":
        # define parameters
        combine_params = params["postprocess_data_params"]
        combine_params["dailyfx"] = True if dailyfx == 'y' else False
        combine_params["econtimes"] = True if dailyfx == 'y' else False

        # run data postprocessings
        run_postprocess(**combine_params)
    
if __name__ == "__main__":
    main_predict_sentiments()