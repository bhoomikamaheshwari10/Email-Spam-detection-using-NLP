# Email Spam Detection using NLP

## Overview

This project focuses on classifying email messages as **Spam** or **Ham (Not Spam)** using Natural Language Processing (NLP) and Machine Learning. It compares the performance of three different text vectorization techniques with Logistic Regression to identify the most effective approach for spam detection.

---

## Problem Statement

Email spam is one of the most common cybersecurity and communication challenges. The objective of this project is to automatically classify incoming emails as spam or legitimate by transforming textual data into numerical features and training a machine learning classifier.

---

## Dataset

The dataset consists of labeled email messages with two classes:

- **Spam (1):** Unwanted promotional or fraudulent emails.
- **Ham (0):** Legitimate emails.

---

## Data Preprocessing

The following preprocessing steps were applied before training the models:

- Removal of special characters and punctuation
- Conversion of text to lowercase
- Tokenization
- Stopword removal
- Lemmatization
- Text reconstruction for feature extraction

---

## Feature Extraction Techniques

Three different NLP techniques were used to convert text into numerical vectors.

### 1. CountVectorizer

CountVectorizer creates a vocabulary from the training corpus and represents each document by the frequency of each word.

**Advantages**
- Simple and fast
- Easy to understand
- Suitable for baseline models

---

### 2. TF-IDF (Term Frequency–Inverse Document Frequency)

TF-IDF assigns weights to words based on their importance within a document while reducing the influence of frequently occurring words.

**Advantages**
- Better representation than simple word counts
- Reduces the impact of common words
- Widely used in text classification tasks

---

### 3. Average Word2Vec

Word2Vec generates dense vector representations for words by learning semantic relationships from the corpus. The document vector is obtained by averaging the vectors of all words in the sentence.

**Advantages**
- Captures semantic similarity between words
- Produces dense feature vectors
- Better contextual representation compared to Bag-of-Words methods

---

## Machine Learning Model

All three feature extraction techniques were evaluated using the same classifier:

- **Logistic Regression**

Logistic Regression is an efficient supervised learning algorithm commonly used for binary classification problems such as spam detection.

---
## Application

A simple and interactive web application was developed using **Streamlit** to demonstrate the spam detection model.

### Features

- Select the feature extraction technique:
  - CountVectorizer
  - TF-IDF
  - Average Word2Vec
- Enter an email message for classification.
- Predict whether the message is **Spam** or **Ham**.
- Compare predictions generated using different feature extraction methods.

## Workflow

1. Load the dataset
2. Perform text preprocessing
3. Generate numerical features using:
   - CountVectorizer
   - TF-IDF
   - Average Word2Vec
4. Split the dataset into training and testing sets
5. Train Logistic Regression models
6. Evaluate model performance using:
   - Accuracy
   - Confusion Matrix
   - Classification Report

---

## Evaluation Metrics

The models were evaluated using:

- Accuracy Score
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

## Libraries Used

- Python
- NumPy
- Pandas
- NLTK
- Scikit-learn
- Gensim
- Regular Expressions (re)

---

## Conclusion

This project demonstrates how different NLP feature extraction techniques influence spam email classification. By comparing CountVectorizer, TF-IDF, and Average Word2Vec using the same Logistic Regression classifier, the project provides insight into the effectiveness of traditional and embedding-based text representations for spam detection.