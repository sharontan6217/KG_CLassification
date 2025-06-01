import pandas as pd
import numpy as np
import random
import sklearn
from sklearn.naive_bayes import CategoricalNB,BernoulliNB,ComplementNB,GaussianNB,MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDClassifier




class myNaiveBayes():
    def categoricalNB(x_train, y_train,x_test):
        model_classification = CategoricalNB()
        model_classification.fit(x_train, y_train)
        score = model_classification.score(x_train,y_train)
        print('score is: ',score)
        y_predict = model_classification.predict(x_test)
        model_name = 'categoricalNB'
        return y_predict, model_name
    def multinomialNB(x_train, y_train,x_test):
        model_classification = MultinomialNB()
        model_classification.fit(x_train, y_train)
        score = model_classification.score(x_train,y_train)
        print('score is: ',score)
        y_predict = model_classification.predict(x_test)
        model_name = 'multinomialNB'
        return y_predict, model_name
    def ComplementNB(x_train, y_train,x_test):
        model_classification = ComplementNB(alpha=0.4)
        model_classification.fit(x_train, y_train)
        score = model_classification.score(x_train,y_train)
        print('score is: ',score)
        y_predict = model_classification.predict(x_test)
        model_name = 'ComplementNB'
        return y_predict, model_name
    def BernoulliNB(x_train, y_train,x_test):
        model_classification = BernoulliNB()
        model_classification.fit(x_train, y_train)
        score = model_classification.score(x_train,y_train)
        print('score is: ',score)
        y_predict = model_classification.predict(x_test)
        model_name = 'BernoulliNB'
        return y_predict, model_name
    def GaussianNB(x_train, y_train,x_test):
        model_classification = GaussianNB()
        model_classification.fit(x_train, y_train)
        score = model_classification.score(x_train,y_train)
        print('score is: ',score)
        y_predict = model_classification.predict(x_test)
        model_name = 'GaussianNB'
        return y_predict, model_name