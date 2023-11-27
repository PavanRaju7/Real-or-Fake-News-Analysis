import pandas as pd
import streamlit as st
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.title('Real or Fake News Prediction')

st.sidebar.header('User Input Parameters')

def user_input_features():
    title = st.sidebar.text_input('Title')
    text = st.sidebar.text_input('Text')
    subject = st.sidebar.text_input('Subject')
    news_data = {'title': [title], 'text': [text], 'subject': [subject]}
    features = pd.DataFrame(news_data)
    return features

df = user_input_features()
st.subheader('User Input parameters')
st.write(df)

# Load the vectorizers
with open('vectorizer_unigrams.pkl', 'rb') as unigram_vectorizer_file:
    vectorizer_unigrams = pickle.load(unigram_vectorizer_file)

with open('vectorizer_bigrams.pkl', 'rb') as bigram_vectorizer_file:
    vectorizer_bigrams = pickle.load(bigram_vectorizer_file)

# Transform the input features using the vectorizers
new_data_tfidf_unigrams = vectorizer_unigrams.transform(df['text'])
new_data_tfidf_bigrams = vectorizer_bigrams.transform(df['text'])

# Concatenate unigrams and bigrams
new_data_tfidf_combined = np.hstack((new_data_tfidf_unigrams.toarray(), new_data_tfidf_bigrams.toarray()))

# Load the RandomForestClassifier model
with open('rf_model.pkl', 'rb') as model_file:
    rf_model = pickle.load(model_file)

# Use the trained model to predict
new_prediction = rf_model.predict(new_data_tfidf_combined)

# Display the prediction result
st.subheader('Predicted Result')
result = 'Fake News' if new_prediction[0] == 0 else 'Real News'
st.write(result)
