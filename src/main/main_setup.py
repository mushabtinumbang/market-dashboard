import os
import sys

from loguru import logger
from datetime import datetime
from src.utilities.config_ import model_path
from transformers import BartForConditionalGeneration, BartTokenizer

def setup_bart_model():
    # Logging
    logger.info("Starting to download BART model and tokenizer ..")

    # Load the tokenizer
    logger.info("Downloading BART Tokenizer ....")
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

    # Load the model
    logger.info("Downloading BART Model ....")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

    # Save the model locally if needed
    logger.info(f"Saving model to {os.path.join(model_path, 'bart-large-cnn')} ....")
    model.save_pretrained(os.path.join(model_path, "bart-large-cnn"))
    tokenizer.save_pretrained(os.path.join(model_path, "bart-large-cnn"))


if __name__ == "__main__":
    setup_bart_model()