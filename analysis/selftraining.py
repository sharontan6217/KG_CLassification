import pandas as pd
import numpy as np
import random
import sklearn
#from sentence_transformers import SentenceTransformer, util
#from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn.semi_supervised import LabelSpreading, SelfTrainingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, RidgeClassifier, RANSACRegressor, PassiveAggressiveRegressor, SGDRegressor, TheilSenRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.svm import LinearSVR, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid, KernelDensity
from sklearn.neural_network import MLPClassifier, BernoulliRBM
from sklearn.naive_bayes import CategoricalNB,BernoulliNB,ComplementNB,GaussianNB,MultinomialNB
from sklearn.linear_model import SGDClassifier

class selfTrainingClassifier():
    def selfTrainingClassifier(x_train, y_train,x_test):
        y_mask = np.random.rand(len(y_train)) < 0.4
        #print(len(y_mask))
        count = 0
        for i in range(len(y_mask)):
            if y_mask[i] == True:
                y_train[i] == -1
                count +=1
        #print(y_mask)
        model_classification = SelfTrainingClassifier(LogisticRegression(random_state=0, verbose=True,solver="liblinear", tol=0.01, C=6000),verbose=True)
        #model_classification = SelfTrainingClassifier(SGDClassifier(alpha=1e-5, penalty="l2", loss="log_loss"),verbose=True)
        #model_classification = SelfTrainingClassifier(LinearSVC(C=10, tol=1e-4,dual=False, max_iter=1000,random_state=0, verbose=True))
        model_classification.fit(x_train, y_train)
        score = model_classification.score(x_train,y_train)
        print('score is: ',score)
        y_predict = model_classification.predict(x_test)
        model_name = 'selfTrainingClassifier'
        return y_predict, model_name
    def LabelSpreading(x_train, y_train,x_test):
        y_mask = np.random.rand(len(y_train)) < 0.4
        #print(len(y_mask))
        for i in range(len(y_mask)):
            if y_mask[i] == True:
                y_train[i] == -1
        print(x_train.shape)
        model_classification = LabelSpreading(alpha=0.9,tol=1e-4)
        #kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}
        model_classification.fit(x_train, y_train)
        score = model_classification.score(x_train,y_train)
        print('score is: ',score)
        y_predict = model_classification.predict(x_test)
        model_name = 'selfTrainingClassifier'
        return y_predict, model_name