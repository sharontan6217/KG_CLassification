import pandas as pd
import numpy as np
import random
import sklearn
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier, BernoulliRBM
from sklearn.linear_model import LogisticRegression, LinearRegression, RidgeClassifier, RANSACRegressor, PassiveAggressiveRegressor, SGDRegressor, TheilSenRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline





class myNN():
    def MLP(x_train, y_train,x_test):  
        #x_train = MinMaxScaler().fit_transform(x_train)
        #x_test = MinMaxScaler().fit_transform(x_test)
        model_classification = MLPClassifier(random_state=0, verbose=True,alpha=0.01,solver='adam',learning_rate_init=0.01,max_iter=1000)
        model_classification.fit(x_train, y_train)
        score = model_classification.score(x_train,y_train)
        print('score is: ',score)
        y_predict = model_classification.predict(x_test)
        model_name='MLP'
        return y_predict,model_name

    def BernoulliRBM(x_train, y_train,x_test, y_test,test_size):  
        x = np.concatenate((x_train,x_test),axis=0)
        y = np.concatenate((y_train,y_test),axis=0)
        model_1 = BernoulliRBM(random_state=0, verbose=True, learning_rate=0.1, n_iter=100, n_components=2000)
        model_2 = LogisticRegression(solver="liblinear", tol=0.01, C=6000)
        #model_2 =  NearestCentroid()
        #model_2 = LinearSVC(C=10, tol=1e-4,dual=False, max_iter=1000,random_state=0, verbose=True)
        model_classification = Pipeline(steps=[("rbm", model_1 ), ("logistic", model_2 )])
        model_classification.fit(x_train, y_train)
        score = model_classification.score(x_train,y_train)
        print('score is: ',score)
        y_predict = model_classification.predict(x_test)
        model_name = 'BernoulliRBM'
        return y_predict, model_name