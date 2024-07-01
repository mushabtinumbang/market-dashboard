import streamlit as st
import numpy as np
import pandas as pd
import streamlit.components.v1 as components

from PIL import Image
from src.utilities.streamlit import *
from streamlit_option_menu import option_menu
from datetime import datetime, timedelta

from src.utilities.config_ import finbert_model_path

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
        df['date'] = pd.to_datetime(df['date'])
        st.session_state.df = df

    else:
         df = st.session_state.df

    if "finbert_model" not in st.session_state:
        finbert_model = get_finbert_model()
        finbert_tokenizer = get_finbert_tokenizer()
        st.session_state.finbert_model = finbert_model
        st.session_state.finbert_tokenizer = finbert_tokenizer

    else:
         finbert_model = st.session_state.finbert_model
         finbert_tokenizer = st.session_state.finbert_tokenizer

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
    "Scrape & Predict News"
    ]

    # Show tabs
    selected = option_menu(None, listTabs, 
        icons=['pie-chart', 'graph-up-arrow'], 
        menu_icon="cast", default_index=0, orientation="horizontal")
    
    # Dashboard
    if selected == "Dashboard":
        # create cols
        title1col, datecol = st.columns((9, 2))

        # Overall Sentiment Part
        with title1col:
            stspace(2)
            st.write("### Overall Sentiment")

        with datecol:
            # Find the minimum date
            min_date = df['date'].min().to_pydatetime()

            # Find the maximum date
            max_date = df['date'].max().to_pydatetime()
            
            # Date Input Widget
            date_chosen = st.date_input("Select Date", value=(max_date, max_date), min_value=min_date, max_value=max_date, format="DD.MM.YYYY", disabled=False, label_visibility="visible")

            # Proccess date
            start_date = date_chosen[0].strftime('%Y-%m-%d')

            # Get end date
            try:
                end_date = date_chosen[1].strftime('%Y-%m-%d')

            # Hadnle errors
            except Exception as e:
                end_date = start_date
            
            # Compute Total Negatives, Neutrals, Positives, and NSS Score
            filtered_df = filter_df_by_date(df, start_date=start_date, end_date=end_date)
            neg, neut, pos, score = calculate_sentiment_metrics(filtered_df)

            # Get Total News by source
            total_dailyfx, total_econtimes, total_financialtimes = get_total_unique_sources(filtered_df)
        
        # Sample value
        sentiment_score = round(score, 3)
        sentiment_label = "Negative" if sentiment_score <= -0.2 else "Neutral" if sentiment_score < 0.2 else "Positive"
        
        # create dashboard cols
        leftcol, midcol = st.columns(2)

        with leftcol:
            # Show Gauge Chart
            st.plotly_chart(create_gauge_chart(sentiment_score))
            
        with midcol:
            stspace(2)
            st.write(f"**Overall Net Sentiment Score (NSS) for a given period is** ***{sentiment_label}***.")
            mid_1, mid_2, mid_3 = st.columns((100, 100, 100))
            
            with mid_1:
                # Give space :)
                stspace(2)

                # Negative Metrics
                st.metric("Negative", f"{neg} News", "-2")

            with mid_2:
                # Give spaazeeee
                stspace(2)

                # Neutral Metrics
                st.metric("Neutral", f"{neut} News", "4")
            
            with mid_3:
                # Give space
                stspace(2)

                # Positive Metrics
                st.metric("Positive", f"{pos} News", "3")

            # Document news sources
            stspace(3)
            st.write(f"###### Total News Per Source ")
            st.write(f"DailyFX : {total_dailyfx}. Economic Times : {total_econtimes}. Financial Times : {total_financialtimes}")
            
        # 2nd Header
        title2col, _ = st.columns((9, 2))

        with title2col:
            # Write Header
            st.write("### Predict News Headline Sentiment")
        
        # Create input widget
        finbert_input = st.text_input("Input News Headline to Predict Sentiment", "")

        if finbert_input:
            finbert_output = predict_with_finbert(finbert_input, finbert_model, finbert_tokenizer)
            st.write(f"###### Predicted Sentiment: {finbert_output[0]}")
        
        # 3rd Header
        newscol1, newscol2 = st.columns((2))

        with newscol1:
            # Write Header
            stspace(2)
            st.write("### News")

            # Loop through filtered_df
            for index, row in filtered_df[:10].iterrows():
                # Get Variables
                title = row['title']
                date = row['date']
                url = row['url']
                category = row['category']
                label = row['label']
                source = row['source']

                color = {
                    "negative": "red",
                    "positive": "limegreen",
                    "neutral": "blue"
                }.get(label.lower(), "black")

                st.markdown(f'<span style="color:{color};">{label.capitalize()}</span>', unsafe_allow_html=True)
                st.markdown(f"##### <a href='{url}' style='color:black; font-size:20px;'>{title}</a>", unsafe_allow_html=True)
                st.write(f'###### Source: {source.capitalize()}. Category: {category.capitalize()}. Date: {date.strftime("%d-%m-%Y")}')
                
                stspace(2)

        
        with newscol2:
            stspace(5)
            # Loop through filtered_df
            for index, row in filtered_df[10:20].iterrows():
                # Get Variables
                title = row['title']
                date = row['date']
                url = row['url']
                category = row['category']
                label = row['label']
                source = row['source']

                color = {
                    "negative": "red",
                    "positive": "limegreen",
                    "neutral": "blue"
                }.get(label.lower(), "black")

                st.markdown(f'<span style="color:{color};">{label.capitalize()}</span>', unsafe_allow_html=True)
                st.markdown(f"##### <a href='{url}' style='color:black; font-size:20px;'>{title}</a>", unsafe_allow_html=True)
                st.write(f'###### Source: {source.capitalize()}. Category: {category.capitalize()}. Date: {date.strftime("%d-%m-%Y")}')
                
                stspace(2)
                    
