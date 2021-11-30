
import pandas as pd
import xml.etree.ElementTree as ET
import re
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

def read_data(path):
    '''
    Lee los datos en formato xml del path data/raw
    '''
    tree = ET.parse(path)
    root = tree.getroot()
    raw_dict = {
    'User': [],
    'Content': [],
    'Date': [],
    'Lang': [],
    'Polarity': [],
    'Type': []
    }

    for i in root.iter('tweet'):
        user = i.find('user').text
        content = i.find('content').text
        date = i.find('date').text
        lang = i.find('lang').text
        polarity = i.find('sentiments').find('polarity').find('value').text
        tweet_type = i.find('sentiments').find('polarity').find('type').text
        
        raw_dict['User'].append(user)
        raw_dict['Content'].append(content)
        raw_dict['Date'].append(date)
        raw_dict['Lang'].append(lang)
        raw_dict['Polarity'].append(polarity)
        raw_dict['Type'].append(tweet_type)
        
    df = pd.DataFrame(raw_dict)
    return df


def signs_tweets(tweet):
    signos = re.compile("(\.)|(\;)|(\:)|(\!)|(\?)|(\¿)|(\@)|(\,)|(\")|(\()|(\))|(\[)|(\])|(\d+)")
    return signos.sub('', tweet.lower())


def remove_links(df):
    return " ".join(['{link}' if ('http') in word else word for word in df.split()])

def remove_stopwords(df):
    spanish_stopwords = stopwords.words('spanish')
    return " ".join([word for word in df.split() if word not in spanish_stopwords])

def spanish_stemmer(x):
    stemmer = SnowballStemmer('spanish')
    return " ".join([stemmer.stem(word) for word in x.split()])

def polaridad_fun(x):
    if x in ('P', 'P+'):
        return 0
    elif x in ('N', 'N+'):
        return 1