import pandas as pd
import click
import os
import sys

from datetime import datetime
from loguru import logger
from src.utilities.config_ import log_path, config_path
from src.utilities.utils import read_yaml
from src.features.scraper_stock import run_scraper_stock
from src.features.summarizer_stock import run_stock_summarizer
from src.features.predict_stock import run_stock_prediction
from src.features.postprocess_stock import run_postprocess_stock

@click.command()
@click.option(
    "--date",
    "-e",
    required=True,
    type=str,
    help="Define 'latest', one specific date '24-01-2024', or date ranges '24-01-2024|26-01-2024'",
)
@click.option(
    "--stock",
    required=True,
    type=str,
    help="Define the stock name. Ex: NVDA, AAPL, INTL, IBM.",
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
    default=["n", "n", "n", "n"],
    show_default=True,
    multiple=True,
    help="example: -p y -p y -p y\n"
    " 1st p = scrape financial news;"
    " 2nd p = summarize news content with BART;"
    " 3rd p = predict sentiments with DistilBERT;"
    " 4th p = prepare data for streamlit;"
)
def main_predict_stock(
    date,
    stock,
    suffix,
    pipeline
):
    container_date = datetime.now().strftime("%Y-%m-%d")
    logger.remove()
    logger.add(
        os.path.join(log_path, "run_predict_stock" + container_date + ".log"),
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
        os.path.join(config_path, "stock_config.yaml"), render=True, suffix=suffix
    )

    if pipeline[0] == "y":
        # define parameters
        scraper_params = params["run_scraper_stock_pipeline_params"][stock]
        scraper_params['date'] = date
        scraper_params['stock'] = stock
        scraper_params['suffix'] = suffix

        # run scraper
        run_scraper_stock(**scraper_params)

    if pipeline[1] == "y":
        # define parameters and run scraper
        summarizer_params = params["run_summarizer_stock_pipeline_params"][stock]
        summarizer_params['stock'] = stock
        run_stock_summarizer(**summarizer_params)

    if pipeline[2] == "y":
        # define parameters and run scraper
        summarizer_params = params["run_sentiment_prediction_stock_pipeline_params"][stock]
        summarizer_params['stock'] = stock
        run_stock_prediction(**summarizer_params)

    if pipeline[3] == "y":
        # define parameters and run scraper
        summarizer_params = params["run_sentiment_postprocessing_pipeline_params"][stock]
        summarizer_params['stock'] = stock
        run_postprocess_stock(**summarizer_params)
        

if __name__ == "__main__":
    main_predict_stock()