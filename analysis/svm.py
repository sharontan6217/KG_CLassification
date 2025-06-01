import pandas as pd
import numpy as np
import random
import sklearn
from sklearn.model_selection import train_test_split
#from sentence_transformers import SentenceTransformer, util
from sklearn.svm import LinearSVR, LinearSVC





class mySVM():
    def LinearSVR(x_train,y_train,x_test):
        model_classification = LinearSVR(tol=1e-4,random_state=0,  verbose=True)
        model_classification.fit(x_train, y_train)
        score = model_classification.score(x_train,y_train)
        print('score is: ',score)
        y_predict = model_classification.predict(x_test)
        model_name = 'LinearSVR'
        return y_predict, model_name
    def LinearSVC(x_train,y_train,x_test):
        model_classification = LinearSVC(C=10, tol=1e-4,dual=False, max_iter=1000,random_state=0, verbose=True)
        model_classification.fit(x_train, y_train)
        score = model_classification.score(x_train,y_train)
        print('score is: ',score)
        y_predict = model_classification.predict(x_test)
        model_name = 'LinearSVC'
        return y_predict, model_name