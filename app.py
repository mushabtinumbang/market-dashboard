import streamlit as st
import numpy as np
import pandas as pd
import streamlit.components.v1 as components
import time

from PIL import Image
from src.utilities.streamlit import *
from streamlit_option_menu import option_menu
from datetime import datetime, timedelta

from src.utilities.config_ import finbert_model_path
from streamlit_calendar import calendar

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

# Global Variables
MAX_NEWS_PER_COL = 10
DATE_VIEW = datetime.today().strftime("%Y-%m-%d") # set default date in the calendar as today.

if True:
    # Greet
    if "greet" not in st.session_state:
        greet()
        st.session_state.greet = True

    # Cache df
    if "df" not in st.session_state:
        df = getdata()
        df['date'] = pd.to_datetime(df['date'])
        st.session_state.df = df

    # Get From Cache
    else:
         df = st.session_state.df

    # Cache Model & Tokenizer
    if "finbert_model" not in st.session_state:
        finbert_model = get_finbert_model()
        finbert_tokenizer = get_finbert_tokenizer()
        st.session_state.finbert_model = finbert_model
        st.session_state.finbert_tokenizer = finbert_tokenizer

    # Get From Cache
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
            # Predict the input
            finbert_output = predict_with_finbert(finbert_input, finbert_model, finbert_tokenizer)

            # Show
            st.write(f"###### Predicted Sentiment: {finbert_output[0]}")
        
        # Setup size formatting
        len_filtered_df = filtered_df.shape[0]
        left_col = min(MAX_NEWS_PER_COL, ((len_filtered_df // 2) + 1) if len_filtered_df % 2 != 0 else (len_filtered_df // 2))
        right_col = min(MAX_NEWS_PER_COL, ((len_filtered_df // 2)))
        
        # 3rd Header
        newscol1, newscol2 = st.columns((2))
        with newscol1:
            # Write Header
            stspace(2)
            st.write("### News")
            
            # Display news in the first column
            display_news(filtered_df, 0, left_col)

        with newscol2:
            # Space
            stspace(5)
            
            # Display news in the second column
            display_news(filtered_df, left_col, left_col + right_col)

    if selected == "Scrape & Predict News":
        # Write Header
        st.write("### Scrape and Predict Sentiments!")

        # Get min and max date for each source
        dailyfx_min, dailyfx_max, econtimes_min, econtimes_max, financialtimes_min, financialtimes_max = get_min_max_date_by_source(df)

        scrapecol1, scrapecol2, scrapecol3 = st.columns((10, 4, 10))

        with scrapecol1:
            # Write Subheader
            st.write("#### Sources")

            # Source columns
            optionscol1, optionscol2, optionscol3 = st.columns(3)

            with optionscol1:
                dailyfx_option = st.checkbox("Scrape DailyFX", value=True)
            
            with optionscol2:
                econtimes_option = st.checkbox("Scrape Economic Times", value=True)
            
            with optionscol3:
                financialtimes_option = st.checkbox("Scrape Financial Times", value=True)

            # Write Subheader
            st.write("#### Select Date")

            # Create calendar based on current df
            calendar = create_calendar(DATE_VIEW, dailyfx_min, dailyfx_max, econtimes_min, econtimes_max, financialtimes_min, financialtimes_max)
            
            # Write Notes
            st.write(f"Note: The colors on the calendar indicates that we have scraped news from a given source for a certain date. It is recommended to scrape news on a date which we haven't scraped before.")

            # Date Processings
            if calendar["callback"] == "dateClick":
                train_date = (datetime.strptime(calendar["dateClick"]["date"], '%Y-%m-%dT%H:%M:%S.%fZ') + pd.Timedelta(days=1)).strftime("%d-%m-%Y")
                
            
            elif calendar["callback"] == "select":
                train_date = f'{(datetime.strptime(calendar["select"]["start"], "%Y-%m-%dT%H:%M:%S.%fZ") + pd.Timedelta(days=1)).strftime("%d-%m-%Y")}|{datetime.strptime(calendar["select"]["end"], "%Y-%m-%dT%H:%M:%S.%fZ").strftime("%d-%m-%Y")}'

            else:
                train_date = "None"

        with scrapecol3:
            # Create a dictionary of options
            options = {
                "DailyFX": dailyfx_option,
                "Economic Times": econtimes_option,
                "Financial Times": financialtimes_option
            }

            # Write Final Parameters Confirmation
            st.write("#### Final Paramaters")
            st.write(f"Date Chosen: {train_date}")
            
            # One-liner to write all True values or None if there aren't any
            selected_sources = [source for source, selected in options.items() if selected]

            st.write(f"Sources Selected: {', '.join(selected_sources) if selected_sources else 'None'}")

            if train_date != "None" and selected_sources:
                # Set Trainable to be used in the button 'disabled' params. Setting this to True will make the button active.
                trainable = True

            else:
                # Set Trainable to be used in the button 'disabled' params. Setting this to False will make the button inactive
                trainable = False

            # Create a button to train
            scrape_button = st.button("Scrape and Predict", type="primary", disabled= not trainable)

            if scrape_button:
                # Log
                st.info(f"Running Web Scraping for {', '.join(selected_sources) if selected_sources else 'None'}", icon="ℹ️")

                # Start Scrape
                scrape_success = run_scrape_streamlit(date=train_date, dailyfx=dailyfx_option, econtimes=econtimes_option, financialtimes=financialtimes_option)

                # Log success
                if scrape_success:
                    st.success('Scraping Success!', icon="✅")

                else:
                    st.error('Scraping Failed', icon="🚨")
    
                # Log
                st.info(f"Predicting Sentiment with FinBert for {', '.join(selected_sources) if selected_sources else 'None'}", icon="ℹ️")
                
                # Start Predidct
                predict_success = run_predict_streamlit(date=train_date, dailyfx=dailyfx_option, econtimes=econtimes_option, financialtimes=financialtimes_option)

                # Log success
                if predict_success:
                    st.success('Predict Sentiment with FinBERT Success!', icon="✅")

                else:
                    st.error('Predict Sentimnet with FinBert Failed', icon="🚨")

                # Another logs
                st.info(f"Starting Postprocessing Steps for {', '.join(selected_sources) if selected_sources else 'None'}", icon="ℹ️")

                # Start Scrape
                postprocess_success = run_postprocess_streamlit(date=train_date, dailyfx=dailyfx_option, econtimes=econtimes_option, financialtimes=financialtimes_option)

                # Log success
                if scrape_success:
                    # Log Postprocessing success
                    st.success('Postprocessing Success!', icon="✅")
                    
                    # Update data
                    df = getdata()
                    df['date'] = pd.to_datetime(df['date'])
                    st.session_state.df = df

                    # Log All done :)
                    st.success('All Done!', icon="✅")
                
                else:
                    # Log Fail
                    st.error('Postprocessing Failed', icon="🚨")


