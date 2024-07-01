import streamlit as st
import time
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
import torch

import src.utilities.utils as utils

from datetime import datetime
from src.utilities.config_ import combined_data_path
from transformers import BertTokenizer, BertForSequenceClassification
from src.utilities.config_ import finbert_model_path

# Chart Variables
quadrant_colors = ["#2bad4e", "#eff229", "#f25829"]  
quadrant_text = ["<b>Negative</b>", "<b>Neutral</b>", "<b>Positive</b>"]
n_quadrants = len(quadrant_colors) 

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

def get_finbert_model():
    finbert_model = BertForSequenceClassification.from_pretrained(finbert_model_path)
    return finbert_model

def get_finbert_tokenizer():
    finbert_tokenizer = BertTokenizer.from_pretrained(finbert_model_path)
    return finbert_tokenizer

def create_gauge_chart(current_value):
    # Configuration for the gauge chart
    plot_bgcolor = "#fff"  # Default white background
    quadrant_colors = [plot_bgcolor, "#2bad4e", "#eff229" , "#f25829"]  # Negative, Neutral, Positive
    quadrant_text = ["", "<b>Positive</b>", "<b>Neutral</b>", "<b>Negative</b>"]
    n_quadrants = len(quadrant_colors) - 1
    
    min_value = -1
    max_value = 1
    hand_length = np.sqrt(2) / 4
    hand_angle = np.pi * (1 - (max(min_value, min(max_value, current_value)) - min_value) / (max_value - min_value))

    fig = go.Figure(
        data=[
            go.Pie(
                values=[0.5] + [0.2, 0.1, 0.2],
                rotation=90,
                hole=0.5,
                marker_colors=quadrant_colors,
                text=quadrant_text,
                textinfo="text",
                hoverinfo="skip",
                sort=False
            ),
        ],
        layout=go.Layout(
            showlegend=False,
            margin=dict(b=0, t=40, l=10, r=10),
            width=400,
            height=400  ,
            paper_bgcolor=plot_bgcolor,
            annotations=[
                go.layout.Annotation(
                    text=f"<b>Net Sentiment Score Value:</b><br>{current_value}",
                    x=0.5, xanchor="center", xref="paper",
                    y=0.25, yanchor="bottom", yref="paper",
                    showarrow=False,
                )
            ],
            shapes=[
                go.layout.Shape(
                    type="circle",
                    x0=0.48, x1=0.52,
                    y0=0.48, y1=0.52,
                    fillcolor="#333",
                    line_color="#333",
                ),
                go.layout.Shape(
                    type="line",
                    x0=0.5, x1=0.5 + hand_length * np.cos(hand_angle),
                    y0=0.5, y1=0.5 + hand_length * np.sin(hand_angle),
                    line=dict(color="#333", width=4)
                )
            ]
        )
    )
    
    return fig

def filter_df_by_date(df, start_date, end_date):
    # Filter the DataFrame based on the date range
    filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    filtered_df = filtered_df.sort_values('date', ascending=False).reset_index(drop=True)
    
    return filtered_df

def calculate_sentiment_metrics(df):
    # count label
    label_counts = df['label'].value_counts()

    # get total for each label 
    total_neutral = label_counts.get('neutral', 0)
    total_positive = label_counts.get('positive', 0)
    total_negative = label_counts.get('negative', 0)
    
    # Weighted Sentiment Score (weights can be adjusted as needed)
    w_p = 1
    w_n = 1
    weighted_sentiment_score = (((w_p * total_positive) - (w_n * total_negative)) / (total_positive + total_negative + total_neutral))
    
    return total_negative, total_neutral, total_positive, weighted_sentiment_score

def get_total_unique_sources(df):

    # count label
    label_counts = df['source'].value_counts()

    # get total for each label 
    total_dailyfx = label_counts.get('dailyfx', 0)
    total_econtimes = label_counts.get('econtimes', 0)
    total_financialtimes = label_counts.get('financialtimes', 0)
    
    return total_dailyfx, total_econtimes, total_financialtimes

def predict_with_finbert(
        text,
        loaded_model,
        loaded_tokenizer,
        batch_size=32
):
    # New text data to predict
    titles = [text]

    # Move model to the appropriate device (e.g., CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded_model.to(device)

    # Initialize an empty list to store predictions
    all_predictions = []

    # Process data in batches
    for i in range(0, len(titles), batch_size):
        batch_titles = titles[i:i + batch_size]
        
        # Tokenize the new text
        inputs = loaded_tokenizer(batch_titles, padding='max_length', truncation=True, max_length=64, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        
        # Make predictions
        loaded_model.eval()
        with torch.no_grad():
            outputs = loaded_model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
        
        # Convert predictions to labels
        int_to_label = {0: 'positive', 1: 'neutral', 2: 'negative'}
        predicted_labels = [int_to_label[pred.item()] for pred in predictions]
        all_predictions.extend(predicted_labels)

    return all_predictions