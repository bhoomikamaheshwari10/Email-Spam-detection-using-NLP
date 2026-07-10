import streamlit as st
import pickle
import numpy as np
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec

# -------------------- Download NLTK data --------------------
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")

# -------------------- Load Models --------------------

# Count Vectorizer
count_model = pickle.load(open("models/count_model.pkl", "rb"))
count_vectorizer = pickle.load(open("models/count_vectorizer.pkl", "rb"))

# TF-IDF
tfidf_model = pickle.load(open("models/tfidf_model.pkl", "rb"))
tfidf_vectorizer = pickle.load(open("models/tfidf_vectorizer.pkl", "rb"))

# Average Word2Vec
avg_model = pickle.load(open("models/avg_word2vec_model.pkl", "rb"))
w2v_model = Word2Vec.load("models/word2vec.model")

lemmatizer = WordNetLemmatizer()

# -------------------- Preprocessing --------------------

def preprocess(text):

    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    text = text.split()

    text = [
        lemmatizer.lemmatize(word)
        for word in text
        if word not in stopwords.words("english")
    ]

    return text


# -------------------- Average Word2Vec --------------------

def avg_word2vec(tokens):

    vectors = []

    for word in tokens:

        if word in w2v_model.wv:

            vectors.append(w2v_model.wv[word])

    if len(vectors) == 0:

        return np.zeros(w2v_model.vector_size)

    return np.mean(vectors, axis=0)


# -------------------- Streamlit UI --------------------

st.set_page_config(
    page_title="Email Spam Detection",
    page_icon="📧"
)

st.title("📧 Email Spam Detection using NLP")

st.write("Select a feature extraction technique and classify an email as Spam or Ham.")

technique = st.selectbox(
    "Choose Feature Extraction Technique",
    [
        "CountVectorizer",
        "TF-IDF",
        "Average Word2Vec"
    ]
)

message = st.text_area("Enter Email Message")

if st.button("Predict"):

    if message.strip() == "":

        st.warning("Please enter an email message.")

    else:

        processed = preprocess(message)

        if technique == "CountVectorizer":

            cleaned = " ".join(processed)

            vector = count_vectorizer.transform([cleaned])

            prediction = count_model.predict(vector)[0]

        elif technique == "TF-IDF":

            cleaned = " ".join(processed)

            vector = tfidf_vectorizer.transform([cleaned])

            prediction = tfidf_model.predict(vector)[0]

        else:

            vector = avg_word2vec(processed)

            prediction = avg_model.predict(vector.reshape(1, -1))[0]

        st.subheader("Prediction")

        if prediction == 1:
            st.error("🚨 Spam Email")
        else:
            st.success("✅ Ham Email")