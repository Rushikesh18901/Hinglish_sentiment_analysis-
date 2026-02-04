A Machine Learning web application that predicts sentiment (Positive / Negative / Neutral) from Hinglish text (Hindi written in English letters).
Built as an NLP + ML project and deployed using Streamlit.

#Features
-Takes Hinglish text input , Text preprocessing ,-TF-IDF vectorization , ML-based sentiment prediction , -Interactive web interface , -User feedback storage (CSV)


#Tech Stack
-Python , Streamlit , Scikit-learn , NumPy , Joblib

#Machine Learning Model
The model was trained using: TF-IDF Vectorizer for text feature extraction , Supervised Machine Learning classifier , Saved using .pkl files and loaded with Joblib




#Project Structure
hinglish-sentiment-analysis/
│
├── app.py                 
├── tfidf_vectorizer.pkl    
├── sentiment_model.pkl     
├── feedback.csv          
├── requirements.txt       
└── README.md          

