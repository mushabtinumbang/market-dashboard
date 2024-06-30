import streamlit as st
import time
import pandas as pd
import numpy as np
import os
from datetime import datetime
import src.utilities.utils as utils
from src.utilities.config_ import combined_data_path

def greet():
    st.toast('Hello!', icon='✅')
    time.sleep(1)
    st.toast('Welcome to the Market Dashboard', icon='✅')

def stspace(num):
    for j in range(num):
        st.write("")

def getdata():
    df = utils.load(os.path.join(combined_data_path, 'combined_data.feather'))
    return df

def ChangeButtonColour(widget_label, font_color, background_color='transparent'):
    htmlstr = f"""
        <script>
            var elements = window.parent.document.querySelectorAll('button');
            for (var i = 0; i < elements.length; ++i) {{ 
                if (elements[i].innerText == '{widget_label}') {{ 
                    elements[i].style.color ='{font_color}';
                    elements[i].style.background = '{background_color}'
                }}
            }}
        </script>
        """
    return (f"{htmlstr}")

def sentiment_category(sentimentlatest):
    if sentimentlatest > 0.1:
        return 'Positive'
    elif sentimentlatest < -0.1:
        return 'Negative'
    else:
        return 'Neutral'