# from __future__ import division
import itertools

# To get rid of those blocks of red warnings
import warnings
warnings.filterwarnings("ignore")

# Standard Imports
import numpy as np
from scipy import stats
import pandas as pd
from math import sqrt
import os
from scipy.stats import spearmanr
from sklearn import metrics
from random import randint


# Vis Imports
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
import plotly.express as px
from pandas.plotting import register_matplotlib_converters
from mpl_toolkits.mplot3d import Axes3D
from wordcloud import WordCloud
from PIL import Image

# Modeling Imports
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.feature_selection import f_regression 
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression, RFE
import sklearn.preprocessing
import statsmodels.api as sm
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import MiniBatchKMeans, KMeans

# NLP Imports
import unicodedata
import re
import json
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords

def top_3_personality_domains(df):
    # 60/20/20 Train, Validate, Test split
    train_val, test = train_test_split(df, stratify=df.type, test_size=.2, random_state=123)
    train, validate = train_test_split(train_val, stratify=train_val.type, test_size=.25, random_state=123)
    # Make the object and fit/transform it
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df.lemmatized)
    # What we are predicting
    y = df.personality_domain
    # Split X and y into train, validate, and test.
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, stratify=y, test_size=.2, random_state=123)
    X_train, X_validate, y_train, y_validate = train_test_split(X_train_val, y_train_val, stratify=y_train_val, test_size=.25, random_state=123)
    # Make train and validate a dataframe
    train = pd.DataFrame(dict(actual=y_train))
    validate = pd.DataFrame(dict(actual=y_validate))
    # Make the object and fit it
    lm = LogisticRegression().fit(X_train, y_train)
    # Make columns for the predictions
    train['bow_predicted_lm'] = lm.predict(X_train)
    validate['bow_predicted_lm'] = lm.predict(X_validate)

    df['personality_domain_encoded'] = df['personality_domain'].apply(filter_3)

    # Make the object
    tfidf = TfidfVectorizer()
    # Fit/Transform
    X = tfidf.fit_transform(df.lemmatized)
    # What we are predicting
    y = df.personality_domain_encoded
    # Split X and y into train, validate, and test 
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, stratify=y, test_size=.2, random_state=123)
    X_train, X_validate, y_train, y_validate = train_test_split(X_train_val, y_train_val, stratify=y_train_val, test_size=.25, random_state=123)

    xgb_clf = XGBClassifier()
    xgb_clf.fit(X_train, y_train)
    score_0 = xgb_clf.score(X_train, y_train)
    score_1 = xgb_clf.score(X_validate, y_validate)
    
    # 60/20/20 Train, Validate, Test split
    train_val2, test2 = train_test_split(df, stratify=df.type, test_size=.2, random_state=123)
    train2, validate2 = train_test_split(train_val2, stratify=train_val2.type, test_size=.25, random_state=123)

    # Make the object
    tfidf = TfidfVectorizer()
    # Fit/Transform
    X = tfidf.fit_transform(train2.lemmatized)

    cluster = KMeans(init = 'k-means++', n_clusters=4, n_init=15, random_state=123)
    cluster.fit(X)
    yhat = cluster.predict(X)
    train2['cluster'] = cluster.labels_
    dummies = pd.get_dummies(train2.cluster, prefix='clust')
    train2 = pd.concat([train2, dummies], axis=1)

    # Transform
    X = tfidf.transform(validate2.lemmatized)

    cluster = KMeans(init = 'k-means++', n_clusters=4, n_init=15, random_state=123)
    cluster.fit(X)
    yhat = cluster.predict(X)
    validate2['cluster'] = cluster.labels_
    dummies = pd.get_dummies(validate2.cluster, prefix='clust')
    validate2 = pd.concat([validate2, dummies], axis=1)

    # Features
    X_train = train2[['clust_0', 'clust_1', 'clust_2', 'clust_3']]
    # What we are predicting
    y_train = train2.personality_domain

    # Features
    X_validate = validate2[['clust_0', 'clust_1', 'clust_2', 'clust_3']]
    # What we are predicting
    y_validate = validate2.personality_domain

    # Make train and validate a dataframe
    train2 = pd.DataFrame(dict(actual=y_train))
    validate2 = pd.DataFrame(dict(actual=y_validate))
    # Make the object and fit it
    MNBclf = MultinomialNB()
    MNBclf.fit(X_train, y_train)
    # Make columns for the predictions
    train2['clust_predicted_MNBclf'] = MNBclf.predict(X_train)
    validate2['clust_predicted_MNBclf'] = MNBclf.predict(X_validate)

    # Print out the results
    print('Top Model per Feature')
    print('-------------')
    print('Bag of Words Logistic Regression Train Accuracy: {:.2%}'.format(accuracy_score(train.actual, train.bow_predicted_lm)))
    print('-------------')
    print('Bag of Words Logistic Regression Validate Accuracy: {:.2%}'.format(accuracy_score(validate.actual, validate.bow_predicted_lm)))
    print('-------------')
    print('XBG TF-IDF Personality Domain Train Accuracy:', round(score_0 * 100, 2), '%')
    print('-------------')
    print('XBG TF-IDF Personality Domain Validate Accuracy:', round(score_1 * 100,2), '%')
    print('-------------')
    print()
    print()
    print('Top Clustering Model')
    print('-------------')
    print('Cluster MultinomialNB Train Accuracy: {:.2%}'.format(accuracy_score(train2.actual, train2.clust_predicted_MNBclf)))
    print('-------------')
    print('Cluster MultinomialNB Validate Accuracy: {:.2%}'.format(accuracy_score(validate2.actual, validate2.clust_predicted_MNBclf)))
    print('-------------')

def personality_domain_test(df):
    # Make the object
    tfidf = TfidfVectorizer()
    # Fit/Transform
    X = tfidf.fit_transform(df.lemmatized)
    # What we are predicting
    y = df.personality_domain_encoded
    # Split X and y into train, validate, and test 
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, stratify=y, test_size=.2, random_state=123)
    X_train, X_validate, y_train, y_validate = train_test_split(X_train_val, y_train_val, stratify=y_train_val, test_size=.25, random_state=123)

    xgb_clf = XGBClassifier()
    xgb_clf.fit(X_train, y_train)
    score = xgb_clf.score(X_test, y_test)
    print('-------------')
    print('XBG TF-IDF Personality Domain Test Accuracy:', round(score * 100, 2), '%')
    print('-------------')