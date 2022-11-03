#imports
import random
import multiprocessing
from multiprocessing import Pool
from datetime import date

# the snscrape.modules.twitter commented out below is necessary to perfom the initial scrape
# and needs to be imported
#import snscrape.modules.twitter as sntwitter

import pandas as pd
import numpy as np
from wordcloud import WordCloud
import re
import matplotlib.pyplot as plt
import seaborn as sns
# plt.style.use('fivethirtyeight')
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import concurrent.futures

from copy import deepcopy
from pprint import pprint



import os

# import spacy
# nlp = spacy.load("en_core_web_sm")

import matplotlib.image as mpimg

from pandas.plotting import table \




import pandas as pd
#import unicode character database
import unicodedata
#import regular expression operations
import re

#import natural language toolkit
import nltk
from nltk.corpus import words
#import our aquire


#import our stopwords list
from nltk.corpus import stopwords
from copy import deepcopy




# Read from the pickle and then display the unique file

def wrangle():
    dataframe=pd.read_pickle("./fivezerominpull.pkl")
    dataframe.name.nunique()



    ptypeurl='https://raw.githubusercontent.com/twitter-personality-predictor/twitter-personality-predictor/main/twitter_handles.csv'

    ptypes=pd.read_csv(ptypeurl);ptypes
    newcols=[]
    for x in ptypes.columns.to_list():
        y=x.lower()
        newcols.append(y)

    ptypes.columns=newcols
    ptypes['handle']=ptypes.twitter
    ptypes.drop(columns='twitter',inplace=True)
    ptypes.name=ptypes.name.str.lower();ptypes

    dataframe.name=dataframe.name.str.lower();dataframe


    a=ptypes.name.values.tolist()
    b=ptypes.type.values.tolist()
    ptypemap=dict(zip(a,b))
    ptypemap
    dataframe.dropna(axis=1,inplace=True)
    dataframe['type']=dataframe.name.map(ptypemap)


    cols=list(set(dataframe.columns)-{'name','handle','type'})
    cols.insert(0,'handle')
    cols.insert(0,'name')
    cols.insert(0,'type')

    dataframe=dataframe[cols]
    dataframe.dropna(axis=0,inplace=True)

    dataframe.columns.to_frame().T
    cols=['type','name','renderedContent','content']
    keep=dataframe[cols];keep

    group1=keep[['type','name','content']].groupby(by=['type','name'])
    lista=set(group1.groups.keys())
    group2=keep[['type','name']].groupby(by=['type'])
    listb=list(set(group2.groups.keys()))
    group3=keep[['name','content']].groupby(by=['name'])   
    indexbyperson={}
    for b in listb:
        g=list(group2.get_group(b).index)
        n=list(group2.get_group(b).name.unique())
        
        ndict={}
        for i in n:
            k=list(group3.get_group(i).index)
            c=list(group3.get_group(i).content)
            ndict.update({i:{'index':k,'content':c}})
        indexbyperson.update({b:{'index':g,'name':ndict}})
    
    more_stopwords = ['like', 'im', 'think', 'dont', 'people', 'know', 'one', 'get', 'really','thing',
                  'would', 'time', 'type', 'make', 'friend', 'ive', 'feel', 'much', 'love',
                 'say', 'way', 'see', 'thing', 'want', 'thing', 'good', 'something', 'lot',
                  'also', 'go', 'always', 'even', 'well', 'someone','https','com','co',',',"'"]



    stops=stopwords.words(['french','german','english','spanish','portuguese'])+ more_stopwords


    pd.to_pickle(stops,'stopwords.pkl')




    def stopfilter(text,stop_words_extend_reduce=["'"]):
        'we use symmetric difference so if a is already in stop words then it will be added to our third set else our third set will be missing it'
        #create oujr english stopwords list
        stops = set(pd.read_pickle('stopwords.pkl'))

    
        stop_words_extend_reduce=set(stop_words_extend_reduce)
        stops=stops.symmetric_difference(stop_words_extend_reduce)

        # stops=(stops|stop_words_extend)-exclude_words
        #another way
        
        filtered=list(filter((lambda x: x not in stops), text.split()))
        filtered=' '.join(filtered)

        return filtered



    def basic_clean(text,regexfilter=r'[^a-z0-9\'\s]'):
        '''   
        Filters out all special characters if you need to edit then supply a new regex filter 
        
        
        
        
        '''
        #make a copy and begin to transform it
        newtext = text.lower()

        #encode into ascii then decode
        newtext = unicodedata.normalize('NFKD', newtext)\
        .encode('ascii', 'ignore')\
        .decode('utf-8')

        #use re.sub to remove special characters
        newtext = re.sub(fr'{regexfilter}', ' ', newtext)

        
        


        return newtext

        
    def lemmatizor(text,regexfilter=r'[^a-z0-9\'\s]'):
        '''    
        
        Takes text, tokenizes it, lemmatizes it
        lemmafiltered=list(filter(lambda x: (len(x)>1 and len(x)<9 and x.isalpha()==True),  lemmatized.split()))
        needs to be commented out after the first run (up to modeling)
        # lemmafiltered=list(filter(lambda x: (len(x)>1 and len(x)<9 and x.isalpha()==True and (x in  total)), lemmatized.split()))
        needs to be un commented commented
        
        
        
        
        
        '''
        total=list(pd.read_pickle('words.pkl'))
        

        #make ready the lemmatizer object
        newtext=tokenizer(text,regexfilter=regexfilter)
        wnl = nltk.stem.WordNetLemmatizer()
        lemmatized=split_apply_join(wnl.lemmatize,newtext)

        # since the average word lenght in English is 4.7 characters we will apply a conservative estimate and drop any word that is larger than 8 characters as it is likely not a word
        # we also recursivley took the set of all words generated then compared that to nltk.corpus.words.words() and used that list as filter this is where total comes from

        # lemmafiltered=list(filter(lambda x: (len(x)>1 and len(x)<9 and x.isalpha()==True and (x in  total)), lemmatized.split()))

        lemmafiltered=list(filter(lambda x: (len(x)>1 and len(x)<9 and x.isalpha()==True),  lemmatized.split()))

        lemmafiltered=' '.join(lemmafiltered)
    
        lemmafiltered=basic_clean(lemmafiltered,regexfilter=regexfilter)

        return lemmafiltered
        
        
    def split_apply_join(funct,listobj):
        'helperfuction letters'

        mapped=map(funct, listobj)
        mapped=list(mapped)
        mapped=''.join(mapped)
    
        return mapped




    def tokenizer(text,regexfilter=r'[^a-z0-9\'\s]'):
        ''' 
        For a large file just save it locally
        
        
        
        
        
        '''
        newtext=basic_clean(text,regexfilter=regexfilter)
        #make ready tokenizer object
        tokenize = nltk.tokenize.ToktokTokenizer()
        #use the tokenizer
        newtext = tokenize.tokenize(newtext, return_str=True)
        return newtext


    num=0
    bigdict={'type':{},'name':{},'stoped_lemma':{},'freq':{}}
    for i in list(indexbyperson.keys()):
        a=indexbyperson.get(i)
        a=a['name']
        for i1 in list(a.keys()):
            listtonormaliz=str(a[i1]['content'])
            newtext=lemmatizor(listtonormaliz,regexfilter=r'[^a-z0-9\'\s]')
            lemma=newtext
        
            stoped=stopfilter(lemma)
            stoped=stoped.replace('https','').replace('com','').replace('co','').replace(',','').strip()
        
            a[i1].update({'stopped_lemma':stoped})         
        
            cool=dict(pd.Series(stoped.split()).value_counts())
            a[i1].update({'word freq':cool})
            bigdict['type'].update({num:i})
            bigdict['stoped_lemma'].update({num:stoped})
            bigdict['freq'].update({num:cool})
            bigdict['name'].update({num:i1})
            num+=1

        

    twitterwordslemma=pd.DataFrame(bigdict)
    twitterwordslemma.columns=['type','name','lemmatized','freq']
   
    twitterwordslemma['type']=twitterwordslemma.type.str.lower()

    pd.to_pickle(twitterwordslemma,'maindalemma.pkl')

    df=pd.read_pickle('maindalemma.pkl')
    df=df[[	'type',	'name',	'lemmatized'	]]
    return df
