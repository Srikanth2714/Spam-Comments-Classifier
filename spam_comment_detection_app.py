import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
import streamlit as st

# Load data
data = pd.read_csv('Youtube01-Psy.csv')
data = data[['CONTENT', 'CLASS']]
data['CLASS'] = data['CLASS'].map({0: 'Not Spam', 1: 'Spam Comment'})

# Prepare data for the model
X = np.array(data['CONTENT'])
y = np.array(data['CLASS'])

# Vectorization
cv = CountVectorizer()
X = cv.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = BernoulliNB()
model.fit(X_train, y_train)

# Streamlit UI
st.title("Spam Comment Classifier")
st.write("Enter a comment to check if it's spam or not:")

user_input = st.text_area("Comment:")

if st.button("Predict"):
    if user_input:
        data_vectorized = cv.transform([user_input]).toarray()
        prediction = model.predict(data_vectorized)
        st.write(f"The comment is: **{prediction[0]}**")
    else:
        st.write("Please enter a comment.")
