import pandas as pd
import numpy as np
import random
import sklearn
from sklearn.tree import tree, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid, KernelDensity, KDTree

import matplotlib.pyplot as plt


class myTree():
    def DecisionTree(x_train, y_train,x_test):  
        model_classification = DecisionTreeClassifier()
        model_tree=model_classification.fit(x_train, y_train)
        score = model_classification.score(x_train,y_train)
        print('score is: ',score)
        y_predict = model_classification.predict(x_test)
        model_name = 'DecisionTree'
        plt.figure(figsize=(12,8))
        tree.plot_tree(model_tree,filled=True)
        plt.show()
        return y_predict, model_name

    def RandomForest(x_train, y_train,x_test):  
        model_classification =  RandomForestClassifier()
        model_classification.fit(x_train, y_train)
        score = model_classification.score(x_train,y_train)
        print('score is: ',score)
        y_predict = model_classification.predict(x_test)
        model_name = 'RandomForest'
        return y_predict, model_name
