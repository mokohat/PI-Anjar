train_path= './data_training_v2.csv'

import csv
import numpy as np
import math
import pandas as pd
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import regex as re
import string
from nltk.tag import CRFTagger
from nltk.tokenize import RegexpTokenizer
import pandas as pd
data_train=pd.read_csv(train_path)
#data_test=pd.read_csv(test_path)
# print(data_train.columns.tolist())

label_train=data_train['Sentiment'].to_numpy()
fitur_train=data_train[['Tweet']]
def preprocess_tweet(text):
    # stopfactory = StopWordRemoverFactory()
    # stopword = stopfactory.create_stop_word_remover()
    # convert text to lower-casek
    # clean_punct = RegexpTokenizer(r'\w+')
    # text_tokenizer=clean_punct.tokenize(text)

    nopunc = text.lower()
    # remove URLs
    nopunc = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', '', text)
    nopunc = re.sub(r'http\S+', '', nopunc)
    # remove number
    nopunc = re.sub(r'\d+', '', nopunc)
    # remove usernames
    nopunc = re.sub('@[^\s]+', '', nopunc)
    # remove the # in #hashtag
    nopunc = re.sub(r'#([^\s]+)', r'\1', nopunc)
    # Check characters to see if they are in punctuation
    nopunc = [char for char in nopunc if char not in string.punctuation]
    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    # stopword = stopfactory.create_stop_word_remover()

    # nopunc = stopwords.remove(nopunc)
    stemfactory = StemmerFactory()
    stemmer = stemfactory.create_stemmer()  
    nopunc = stemmer.stem(nopunc)
    clean_punct = RegexpTokenizer(r'\w+') 
    preprocess =clean_punct.tokenize(nopunc)
    #stopwords = nltk.corpus.stopwords.words('english')
    stopwords_clean= ["i",
 "me",
 "my",
 "myself",
 "we",
 "our",
 "ours",
 "ourselves",
 "you",
 "you're",
 "you've",
 "you'll",
 "you'd",
 "your",
 "yours",
 "yourself",
 "yourselves",
 "he",
 "him",
 "his",
 "himself",
 "she",
 "she's",
 "her",
 "hers",
 "herself",
 "it",
 "it's",
 "its",
 "itself",
 "they",
 "them",
 "their",
 "theirs",
 "themselves",
 "what",
 "which",
 "who",
 "whom",
 "this",
 "that",
 "that'll",
 "these",
 "those",
 "am",
 "is",
 "are",
 "was",
 "were",
 "be",
 "been",
 "being",
 "have",
 "has",
 "had",
 "having",
 "do",
 "does",
 "did",
 "doing",
 "a",
 "an",
 "the",
 "and",
 "but",
 "if",
 "or",
 "because",
 "as",
 "until",
 "while",
 "of",
 "at",
 "by",
 "for",
 "with",
 "about",
 "against",
 "between",
 "into",
 "through",
 "during",
 "before",
 "after",
 "to",
 "from",
 "again",
 "further",
 "then",
 "once",
 "here",
 "there",
 "when",
 "where",
 "why",
 "how",
 "all",
 "any",
 "both",
 "each",
 "few",
 "more",
 "most",
 "other",
 "some",
 "such",
 "only",
 "own",
 "same",
 "so",
 "than",
 "too",
 "very",
 "s",
 "t",
 "will",
 "just",
 "now",
 "d",
 "ll",
 "m",
 "o",
 "re"
 "many",
 "ve"]
    text_tokenizer= [word for word in preprocess if word not in (stopwords_clean)]
    text_tokenizer= ' '.join(text_tokenizer)
    return text_tokenizer
    
data_train['Clean_Tweet']= fitur_train['Tweet'].apply(lambda x: preprocess_tweet(x))
data_train.to_csv('./test_baru.csv')
# print(data_train)
