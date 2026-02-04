import streamlit as st
import joblib
import re
import numpy as np

# -------------------------
# Load saved model & vectorizer
# -------------------------
vectorizer = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("sentiment_model.pkl")

# -------------------------
# Label mapping
# -------------------------
label2id = {"Positive":0, "Negative":1, "Neutral":2}
id2label = {v:k for k,v in label2id.items()}

# -------------------------
# Text cleaning
# -------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\u0900-\u097F a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# -------------------------
# Streamlit UI
# -------------------------
st.title("Hindi / Hinglish Sentiment Predictor")
sentence = st.text_area("Enter your feedback here:")

if st.button("Predict"):
    if sentence.strip() == "":
        st.warning("Please enter a sentence.")
    else:
        # Clean text
        cleaned = clean_text(sentence)
        X_input = vectorizer.transform([cleaned])
        
        # Predict class
        prediction_id = model.predict(X_input)[0]
        prediction_label = id2label[prediction_id]
        
        # Predict probability
        probs = model.predict_proba(X_input)[0]
        confidence = np.max(probs) * 100
        
        # Display results
        st.success(f"Predicted Sentiment: {prediction_label}")
        st.info(f"Confidence: {confidence:.2f}%")
