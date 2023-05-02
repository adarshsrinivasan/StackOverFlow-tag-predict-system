#!/usr/bin/env python

import argparse
import pandas as pd
import nltk
import joblib
import re

from flask import Flask, request, jsonify
from google.cloud import storage
from nltk import ToktokTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from nltk.corpus import stopwords

nltk.data.path.append('/bda-stackoverflow-tags')
joblib_model_name = "stack_overflow_model.joblib"
joblib_vectorizer_name = "stack_overflow_vectorizer.joblib"
joblib_binarizer_name = "stack_overflow_binarizer.joblib"
model = None
vectorizer = None
binarizer = None
port = 5000

# Initialize Flask app
app = Flask(__name__)

# Set up lemmatizer and TF-IDF vectorizer
lemmatizer = WordNetLemmatizer()
tfidf = TfidfVectorizer()

punct = '!"$%&\'#()*+,./:;<=>?@[\\]^_`{|}~'
token = ToktokTokenizer()
lemma = WordNetLemmatizer()


def clean_text(text):
    text = text.lower()
    text = text.strip(' ')
    return text


def strip_list_noempty(mylist):
    newlist = (item.strip() if hasattr(item, 'strip') else item for item in mylist)
    return [item for item in newlist if item != '']


def clean_punct(text):
    global token
    words = token.tokenize(text)
    tokens = [token for token in words if not re.match(re.compile('[%s]' % re.escape(punct)), token)]
    filtered_list = strip_list_noempty(tokens)
    return ' '.join(map(str, filtered_list))


def lemmatizeWords(text):
    global token, lemma
    words = token.tokenize(text)
    listLemma = []
    for w in words:
        x = lemma.lemmatize(w, pos="v")
        listLemma.append(x)
    return ' '.join(map(str, listLemma))


def stopWordsRemove(text):
    global token
    stop_words = set(stopwords.words("english"))
    words = token.tokenize(text)
    filtered = [w for w in words if not w in stop_words]
    return ' '.join(map(str, filtered))


def load_and_initialize_model(project, bucket_name):
    global model, vectorizer, binarizer
    client = storage.Client(project=project)
    bucket = client.get_bucket(bucket_name)

    model_blob = bucket.blob(joblib_model_name)
    file_name = f"/tmp/{joblib_model_name}"
    model_blob.download_to_filename(file_name)
    model = joblib.load(file_name)

    vectorizer_blob = bucket.blob(joblib_vectorizer_name)
    file_name = f"/tmp/{joblib_vectorizer_name}"
    vectorizer_blob.download_to_filename(file_name)
    vectorizer = joblib.load(file_name)

    binarizer_blob = bucket.blob(joblib_binarizer_name)
    file_name = f"/tmp/{joblib_binarizer_name}"
    binarizer_blob.download_to_filename(file_name)
    binarizer = joblib.load(file_name)


# Define API endpoint
@app.route('/api/predict_tags', methods=['POST'])
def predict_tags():
    global model, vectorizer, binarizer
    if model is None or vectorizer is None or binarizer is None:
        return jsonify({f"Model is not initialized."}), 500

    # Get input question from user
    data = request.get_json()
    question = data['question']

    data = [[question]]
    new_df = pd.DataFrame(data, columns=['Question'])

    new_df['Question'] = new_df['Question'].apply(lambda x: clean_text(x))
    new_df['Question'] = new_df['Question'].apply(lambda x: clean_punct(x))
    new_df['Question'] = new_df['Question'].apply(lambda x: lemmatizeWords(x))
    new_df['Question'] = new_df['Question'].apply(lambda x: stopWordsRemove(x))

    X = new_df['Question']

    X_tfidf = vectorizer.transform(X)

    pred = model.predict(X_tfidf)
    prediction = binarizer.inverse_transform(pred)

    # Return predicted tags
    return jsonify({'tags': prediction})


@app.route('/api/ping', methods=['GET'])
def ping():
    return "pong"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--project_id",
        type=str,
        required=True,
        help="Project to use for creating resources.",
    )
    parser.add_argument(
        "--gcs_bucket", help="Bucket to upload Pyspark file to", required=True
    )

    args = parser.parse_args()

    load_and_initialize_model(args.project_id, args.gcs_bucket)
    app.run(debug=True, host='0.0.0.0', port=port)
