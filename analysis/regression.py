import pandas as pd
import numpy as np
import random
import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn.linear_model import LogisticRegression, LinearRegression, RidgeClassifier, RANSACRegressor, PassiveAggressiveRegressor, SGDRegressor, TheilSenRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import SGDClassifier



class myRegression():
    def LogisticRegression(x_train, y_train,x_test):
        model_classification = LogisticRegression(random_state=0, verbose=True,solver="liblinear", tol=0.01, C=6000)
        model_classification.fit(x_train, y_train)
        score = model_classification.score(x_train,y_train)
        print('score is: ',score)
        y_predict = model_classification.predict(x_test)
        model_name = 'LogisticRegression'
        return y_predict, model_name
    def LinearRegression(x_train, y_train,x_test):
        model_classification = LinearRegression()
        model_classification.fit(x_train, y_train)
        score = model_classification.score(x_train,y_train)
        print('score is: ',score)
        y_predict = model_classification.predict(x_test)
        model_name = 'LinearRegression'
        return y_predict, model_name
    def RidgeClassifier(x_train, y_train,x_test):
        model_classification = RidgeClassifier(tol=2e-4, solver="sparse_cg",random_state=0)
        model_classification.fit(x_train, y_train)
        score = model_classification.score(x_train,y_train)
        print('score is: ',score)
        y_predict = model_classification.predict(x_test)
        model_name = 'RidgeClassifier'
        return y_predict, model_name
    def RANSACRegressor(x_train, y_train,x_test):
        model_classification = RANSACRegressor(min_samples=len(x_train),random_state=0)
        model_classification.fit(x_train, y_train)
        score = model_classification.score(x_train,y_train)
        print('score is: ',score)
        y_predict = model_classification.predict(x_test)
        model_name = 'RANSACRegressor'

        return y_predict, model_name
    def PassiveAggressiveRegressor(x_train, y_train,x_test):
        model_classification = PassiveAggressiveRegressor(random_state=0, verbose=True)
        model_classification.fit(x_train, y_train)
        score = model_classification.score(x_train,y_train)
        print('score is: ',score)
        y_predict = model_classification.predict(x_test)
        model_name = 'PassiveAggressiveRegressor'
        return y_predict, model_name
    def SGDRegressor(x_train, y_train,x_test):
        model_classification = SGDRegressor(alpha=1e-2, penalty="l2", loss="squared_error",random_state=0, verbose=True)
        model_classification.fit(x_train, y_train)
        score = model_classification.score(x_train,y_train)
        print('score is: ',score)
        y_predict = model_classification.predict(x_test)
        model_name = 'SGDRegressor'
        return y_predict, model_name
    def SGDClassifier(x_train, y_train,x_test):
        model_classification = SGDClassifier(alpha=2e-2, penalty="l2", loss="log_loss",random_state=0, verbose=True)
        model_classification.fit(x_train, y_train)
        score = model_classification.score(x_train,y_train)
        print('score is: ',score)
        y_predict = model_classification.predict(x_test)
        model_name = 'SGDClassifier'
        return y_predict, model_name
    def TheilSenRegressor(x_train, y_train,x_test):
        model_classification = TheilSenRegressor(random_state=0, verbose=True)
        model_classification.fit(x_train, y_train)
        score = model_classification.score(x_train,y_train)
        print('score is: ',score)
        y_predict = model_classification.predict(x_test)
        model_name = 'TheilSenRegressor'
        return y_predict, model_name