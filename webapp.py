"""
# My first app
Here's our first attempt at using data to create a table:
"""

import streamlit as st
import pandas as pd
from prediction import prediction

DATASET_FILE = "./cleaned_dataset.csv"
dataset_df = pd.read_csv(DATASET_FILE)

n = st.slider('Number of topics to show', 1, 15)

text = st.text_input('Review', 'Type here...')

if st.button('Predict'):
    topics = prediction(text, n)

    if type(topics) is str:
        st.write(topics)
    else:
        for i, topic in enumerate(topics):
            st.write(f"{i + 1} : {topic}")
