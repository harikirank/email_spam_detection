from flask import Flask,request,jsonify
import numpy as np
import pandas as pd
import string
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from os.path import dirname, join
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle

classifier = pickle.load(open('model.pickel','rb'))

app = Flask(__name__)
@app.route('/')
def index():
    return "Hello world"
@app.route('/predict/<email>',methods=['GET'])
def predict(email):
    count_vectorizer_fitted = pickle.load(open('fitted_vector.pickel','rb'))
    count_vectorizer_trasnformer = pickle.load(open('transformed_vector.pickel','rb'))

    email_message = email.split("=")[1]

    email_count_vector = count_vectorizer_fitted.transform(pd.DataFrame(data={"text": [email_message]})['text'])
    result = classifier.predict(email_count_vector)
    return jsonify({'isSpam':str(result)[1]})

# 1.4 Cleaning data from punctuation and stopwords and then tokenizing it into words (tokens)
def process(text):
    # Stop words from the nltk dataset
    stop_words = []
    with open(join(dirname(__file__), "stopwords/english.txt")) as f:
        lines = f.readlines()
        for line in lines:
            stop_words.append(line)

    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    clean = [word for word in nopunc.split() if word.lower() not in stop_words]
    return clean

if __name__ == '__main__':
    app.run(debug=True)
