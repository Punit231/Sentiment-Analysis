
import streamlit as st
import joblib
import pandas as pd
import random

# Load model and data
model = joblib.load('sentiment_model.pkl')
df = pd.read_csv('sample_movie_reviews.csv')

# Title
st.title("🎬 Sentiment Analysis - Movie Review Classifier")
st.write("Enter a movie review below or insert a random one from the dataset to analyze its sentiment.")

# Session state for review text
if "text" not in st.session_state:
    st.session_state["text"] = ""

# Insert random example
if st.button("Insert Random Example"):
    example = df.sample(1).iloc[0]["review"]
    st.session_state["text"] = example

# Text input area
user_input = st.text_area("Your Review", height=150, value=st.session_state["text"])

# Predict sentiment
if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        prediction = model.predict([user_input])[0]
        emoji = "😊" if prediction == "pos" else "😞"
        st.success(f"**Sentiment:** `{prediction.upper()}` {emoji}")
