
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Sample training for demo
reviews = [
    "The food was amazing and the service was excellent!",
    "I did not like the taste of the dish."
]
labels = ["Positive", "Negative"]

vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(reviews)
model = LogisticRegression()
model.fit(X, labels)

st.title("Restaurant Review Sentiment Analyzer")
review_input = st.text_input("Enter a review:")
if review_input:
    review_vec = vectorizer.transform([review_input])
    prediction = model.predict(review_vec)
    st.write("Predicted Sentiment:", prediction[0])
