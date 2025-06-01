import pandas as pd
import numpy as np
import tf_keras
import random
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer, HashingVectorizer
from transformers import AutoTokenizer
import gensim
from gensim.models import Word2Vec
import tf_keras as keras
import utils
from utils import rel_utils
from utils.rel_utils import preprocess


class MLTokenizer():
    def word2vec(input):
        if type(input)!=str:
            input_=[]
            for text in input:
                text_ = preprocess.dataClean(text)
                input_.append(text_)
            #print(input_)
            input_vectorized = Word2Vec(input_,min_count=1,window=5,vector_size=300).__getitems__
        else:
            input = preprocess.dataClean(input)
            input_vectorized = Word2Vec([input_],min_count=1,window=5,vector_size=300).__getitems__
        input_vectorized = input_vectorized.toarray()
        return input_vectorized
    def text2vec(input):
        vectorizer = tf_keras.layers.TextVectorization()
        if type(input)!=str:
            input_=[]
            for text in input:
                text_ = preprocess.dataClean(text)
                input_.append(text_)
            #print(input_)
            input_vectorized = vectorizer(input_)
        else:
            input = preprocess.dataClean(input)
            input_vectorized = vectorizer([input])
        input_vectorized = input_vectorized.toarray()
        return input_vectorized
    def tokenizerTFID(input):
        vectorizer = TfidfVectorizer()
        if type(input)!=str:
            input_=[]
            for text in input:
                text_ = preprocess.dataClean(text)
                input_.append(text_)
            #print(input_)
            input_vectorized = vectorizer.fit_transform(input_)
        else:
            input = preprocess.dataClean(input)
            input_vectorized = vectorizer.fit_transform([input])
        input_vectorized = input_vectorized.toarray()
        return input_vectorized
    def tokenizerCount(input):
        vectorizer = CountVectorizer()
        if type(input)!=str:
            input_=[]
            for text in input:
                text_ = preprocess.dataClean(text)
                input_.append(text_)
            #print(input_)
            input_vectorized = vectorizer.fit_transform(input_)
        else:
            input = preprocess.dataClean(input)
            input_vectorized = vectorizer.fit_transform([input])
        input_vectorized = input_vectorized.toarray()
        return input_vectorized
    def tokenizerHashing(input):
        vectorizer = HashingVectorizer()
        if type(input)!=str:
            input_=[]
            for text in input:
                text_ = preprocess.dataClean(text)
                input_.append(text_)
            #print(input_)
            input_vectorized = vectorizer.fit_transform(input_)
        else:
            input = preprocess.dataClean(input)
            input_vectorized = vectorizer.fit_transform([input])
        input_vectorized = input_vectorized.toarray()
        return input_vectorized
    def tokenizerHybrid(input):
        vectorizer = CountVectorizer()
        transformer = TfidfTransformer()
        if type(input)!=str:
            input_=[]
            for text in input:
                text_ = preprocess.dataClean(text)
                input_.append(text_)
            #print(input_)
            input_vectorized_ = vectorizer.fit_transform(input_)
            input_vectorized = transformer.fit_transform(input_vectorized_)
        else:
            input = preprocess.dataClean(input)
            input_vectorized_ = vectorizer.fit_transform(input_)
            input_vectorized = transformer.fit_transform(input_vectorized_)
        input_vectorized = input_vectorized.toarray()
        return input_vectorized
