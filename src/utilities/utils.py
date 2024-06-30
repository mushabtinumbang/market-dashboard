import pandas as pd
import numpy as np
import pickle
import feather
import gzip
import yaml
import os 

from datetime import datetime
from jinja2 import Template

def format_dates(date_input):
    """
    This function takes a date input which could be 'latest', a single date,
    or a date range, and returns a list of dates in the datetime format.
    """
    today_date = datetime.today()

    if date_input.lower() == "latest":
        return [today_date]
    
    date_parts = date_input.split("|")
    
    if len(date_parts) == 1:
        return [datetime.strptime(date_parts[0], '%d-%m-%Y')]
    
    elif len(date_parts) == 2:
        return [
            datetime.strptime(date_parts[0], '%d-%m-%Y'),
            datetime.strptime(date_parts[1], '%d-%m-%Y')
        ]
    
    else:
        raise ValueError("Invalid date input format")
    
def date_to_integer(dates):
    """
    This function converts datetime objects to integer dates based on a reference date.
    Reference date: 43466 represents 01 January 2019.
    """
    reference_date = datetime(2019, 1, 1)
    reference_integer = 43466
    
    integer_dates = [(reference_integer + (date - reference_date).days) for date in dates]
    return integer_dates

def dates_to_string(dates_list):
    """
    This function takes a list of date strings and concatenates them into a single string.
    If there is only one date, it returns the date string.
    If there are multiple dates, it joins them with an underscore.
    """
    if len(dates_list) == 1:
        return dates_list[0]
    else:
        return '_'.join(dates_list)
    
def predict(
        df,
        loaded_svm_model,
        loaded_vectorizer
):
    # New text data to predict
    titles = list(df['title'])

    # Preprocess the new text data
    new_texts_tfidf = loaded_vectorizer.transform(titles)

    # Predict the labels for the new text data
    predictions = loaded_svm_model.predict(new_texts_tfidf)
    df["label"] = predictions

    return df

def save(data, filename):
    folders = os.path.dirname(filename)
    if folders:
        os.makedirs(folders, exist_ok=True)

    fl = filename.lower()
    if fl.endswith(".gz"):
        if fl.endswith(".feather.gz") or fl.endswith(".fthr.gz"):
            # Since feather doesn't support writing to the file handle, we
            # can't easily point it to gzip.
            raise NotImplementedError(
                "Saving to compressed .feather not currently supported."
            )
        else:
            fp = gzip.open(filename, "wb")
            pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        if fl.endswith(".feather") or fl.endswith(".fthr"):
            if str(type(data)) != "<class 'pandas.core.frame.DataFrame'>":
                raise TypeError(
                    ".feather format can only be used to save pandas "
                    "DataFrames"
                )
            feather.write_dataframe(data, filename)
        else:
            fp = open(filename, "wb")
            pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)


def load(filename):
    """
    Loads data saved with save() (or just normally saved with pickle).
    Autodetects gzip if filename ends in '.gz'
    Also reads feather files denoted .feather or .fthr.

    Parameters
    ----------
    filename -- String with the relative filename of the pickle/feather
    to load.
    """
    fl = filename.lower()
    if fl.endswith(".gz"):
        if fl.endswith(".feather.gz") or fl.endswith(".fthr.gz"):
            raise NotImplementedError("Compressed feather is not supported.")
        else:
            fp = gzip.open(filename, "rb")
            return pickle.load(fp)
    else:
        if fl.endswith(".feather") or fl.endswith(".fthr"):
            import feather

            return feather.read_dataframe(filename)
        else:
            fp = open(filename, "rb")
            return pickle.load(fp)


def read_yaml(filename, render=False, **kwargs):
    """
    Read yaml configuation and returns the dict

    Parameters
    ----------
    filename: string
        Path including yaml file name
    render: Boolean, default = False
        Template rendering
    **kwargs:
        Template render args to be passed
    """
    if render:
        yaml_text = Template(open(filename, "r").read())
        yaml_text = yaml_text.render(**kwargs)
        config = yaml.safe_load(yaml_text)
    else:
        with open(filename) as f:
            config = yaml.safe_load(f)

    return config


