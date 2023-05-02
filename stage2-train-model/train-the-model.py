#!/usr/bin/env python
# Copyright 2018 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pickle

import pandas as pd
import io
import nltk
import re

from nltk.tokenize import ToktokTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from google.cloud import storage

nltk.data.path.append('/bda-stackoverflow-tags')

project = "bdastackoverflow"
bucket_name = "bda-dataproc"
prefix = "query-result"
joblib_model_name = "stack_overflow_model.joblib"
joblib_vectorizer_name = "stack_overflow_vectorizer.joblib"
joblib_binarizer_name = "stack_overflow_binarizer.joblib"

os.environ.setdefault("GCLOUD_PROJECT", project)

storage_client = storage.Client()
blobs = storage_client.list_blobs(bucket_name, prefix=prefix)

# Initialize an empty list to store dataframes
dfs = []

# Loop through all the CSV files and read them into a dataframe
for blob in blobs:
    if blob.name.endswith('.csv'):
        # Read CSV from GCS into a pandas dataframe
        blob_content = blob.download_as_string()
        df = pd.read_csv(io.StringIO(blob_content.decode('utf-8')))

        # Append dataframe to list
        dfs.append(df)

# Concatenate all dataframes into a single dataframe
df = pd.concat(dfs, ignore_index=True)

df['tags'] = df['tags'].str.split('|').apply(list)

q = set()
for sublist in df['tags'].values:
    q.update(sublist)
all_tags_list = [item for row in df['tags'] for item in row]
tags_counts = nltk.FreqDist(all_tags_list)
_50_most_common = tags_counts.most_common(50)
most_common_tags = [tag[0] for tag in _50_most_common]
tag_freq = [tags_counts[tag] for tag in most_common_tags]


def most_common(tags):
    tags_to_send = []
    for t in tags:
        if t in most_common_tags:
            tags_to_send.append(t)
    return tags_to_send


df['tags'] = df['tags'].apply(lambda x: most_common(x))
df['tags'] = df['tags'].apply(lambda x: x if len(x) > 0 else None)
df.dropna(subset='tags', inplace=True)

df = df.rename(columns={'body': 'Body'})
df = df.rename(columns={'title': 'Title'})
df = df.rename(columns={'tags': 'Tags'})

token = ToktokTokenizer()
punct = '!"$%&\'#()*+,./:;<=>?@[\\]^_`{|}~'
lemma = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


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
    words = token.tokenize(text)
    listLemma = []
    for w in words:
        x = lemma.lemmatize(w, pos="v")
        listLemma.append(x)
    return ' '.join(map(str, listLemma))


def stopWordsRemove(text):
    stop_words = set(stopwords.words("english"))
    words = token.tokenize(text)
    filtered = [w for w in words if not w in stop_words]
    return ' '.join(map(str, filtered))


df['Body'] = df['Body'].apply(lambda x: clean_text(x))
df['Body'] = df['Body'].apply(lambda x: clean_punct(x))
df['Body'] = df['Body'].apply(lambda x: lemmatizeWords(x))
df['Body'] = df['Body'].apply(lambda x: stopWordsRemove(x))

df['Title'] = df['Title'].apply(lambda x: str(x))
df['Title'] = df['Title'].apply(lambda x: clean_text(x))
df['Title'] = df['Title'].apply(lambda x: clean_punct(x))
df['Title'] = df['Title'].apply(lambda x: lemmatizeWords(x))
df['Title'] = df['Title'].apply(lambda x: stopWordsRemove(x))

df['Combined_text'] = df['Title'] + ' ' + df['Body']

no_topics = 20

y_label = df['Tags']
X_C3 = df['Combined_text']

multilabel_binarizer = MultiLabelBinarizer()
y_bin = multilabel_binarizer.fit_transform(y_label)

vectorizer_X_1 = TfidfVectorizer(analyzer='word',
                                 min_df=0.0,
                                 max_df=1.0,
                                 strip_accents=None,
                                 encoding='utf-8',
                                 preprocessor=None,
                                 token_pattern=r"(?u)\S\S+",
                                 max_features=1000)

X_C3_tfidf = vectorizer_X_1.fit_transform(X_C3)

X_train, X_test, y_train, y_test = train_test_split(X_C3_tfidf, y_bin, test_size=0.2, random_state=0)  # Do 80/20 split

classifier = LogisticRegression(random_state=42, max_iter=1000)

clf = OneVsRestClassifier(classifier, verbose=0, n_jobs=-1)
clf.fit(X_train, y_train)

client = storage.Client(project=project)
bucket = client.get_bucket(bucket_name)

# Save trained to GCS bucket
model_blob = bucket.blob(joblib_model_name)
model_blob.upload_from_string(pickle.dumps(clf))

vectorizer_blob = bucket.blob(joblib_vectorizer_name)
vectorizer_blob.upload_from_string(pickle.dumps(vectorizer_X_1))

binarizer_blob = bucket.blob(joblib_binarizer_name)
binarizer_blob.upload_from_string(pickle.dumps(multilabel_binarizer))
