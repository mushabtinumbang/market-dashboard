import streamlit as st
import numpy as np
import pandas as pd
import streamlit.components.v1 as components
from PIL import Image
from src.utilities.streamlit import *
from streamlit_option_menu import option_menu
from datetime import datetime, timedelta


# Set Layout Config
st.set_page_config(page_title="Market Dashboard", page_icon=":chart_with_upwards_trend:",layout="wide")
     
# Add Font
streamlit_style = """
			<style>
			@import url('https://fonts.cdnfonts.com/css/neue-haas-grotesk-display-pro');    

			html, body, [class*="css"]  {
			font-family: 'neue haas grotesk display pro';
			}
			</style>
			"""

# Apply Font
st.markdown(streamlit_style, unsafe_allow_html=True)

if True:
    # Greet
    if "greet" not in st.session_state:
        greet()
        st.session_state.greet = True

    if "df" not in st.session_state:
        df = getdata()
        st.session_state.df = df

    else:
         df = st.session_state.df

    # Make a Centered Text
    col1, col2, col3 = st.columns((14, 6, 6))

    # Add Image
    with col3:
        st.image(Image.open("./img/logo-unpad.png"))

    # Add Header Text
    with col1:
        stspace(1)
        st.write("## Market Dashboard")

    # Add whitespace
    stspace(3)

    # Generate Tabs
    listTabs = [
    "Dashboard",
    "Scrape & Train News"
    ]

    # Show tabs
    selected = option_menu(None, listTabs, 
        icons=['pie-chart', 'graph-up-arrow'], 
        menu_icon="cast", default_index=0, orientation="horizontal")
    
    # Dashboard
    if selected == "Dashboard":
        spacecol, forexcol, volcol, datecol = st.columns((50, 6, 6, 12))
        marketcol0, marketcol1 = st.columns((2, 3))
        
        with marketcol0:
            st.write("### Overall Sentiment")
