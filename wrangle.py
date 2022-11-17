#import pandas and numpy to interact with data as dataframe
import pandas as pd
import numpy as np
#import unicode character database
import unicodedata
#import regular expression operations
import re
#import natural language toolkit
import nltk
from nltk.corpus import words
#import our stopwords list
from nltk.corpus import stopwords
#import sentiment analysis
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split

def wrangle():
    """This function is all encompassing to acquire and clean/prepare the data. there are 5 functions that are embedded inside this function that are used to return the personality information in a DataFrame"""
    df=pd.read_pickle('tweet_data.pkl')
    c=0
    words_per_tweet = []
    for celeb in df.docs:
        words = 0
        tweets = 0
        for tweet in celeb:
            tweets += 1
            words += len(tweet.split(' '))
        # print(f'Celeb: {c}, Tweets: {tweets}, Words: {words}')
        words_per_tweet.append(round((words/tweets),2))
        c +=1
    df['words_per_tweet'] = words_per_tweet
    char_per_tweet = []
    for celeb in df.docs:
        tweets = 0
        chars = 0
        for tweet in celeb:
            tweets += 1
            chars += len(tweet)
        char_per_tweet.append(round((chars/tweets),2))
    df['char_per_tweet'] = char_per_tweet

    s = nltk.sentiment.SentimentIntensityAnalyzer()
    df['sentiment'] = df.lemmatized.apply(lambda doc: s.polarity_scores(doc)['compound'])

    return df
    
def split_data(df):
    # create train and test (80/20 split) from the orginal dataframe
    train, test = train_test_split(df, test_size=0.2, random_state=123)
    # create train and validate (75/25 split) from the train dataframe
    train, val = train_test_split(train, test_size=.25, random_state=123)
    
    return train, val, test