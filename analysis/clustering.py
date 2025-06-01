import pandas as pd
import numpy as np
import random
import sklearn
from sklearn.model_selection import train_test_split
#from sentence_transformers import SentenceTransformer, util
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier, BernoulliRBM
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn.svm import LinearSVR, LinearSVC
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid, KernelDensity, KDTree
from sklearn.cluster import DBSCAN
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from scipy.spatial.distance import sqeuclidean,jaccard,canberra,cdist
from sklearn.decomposition import PCA, KernelPCA, IncrementalPCA
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid, KernelDensity, KDTree
import math



class myClustering():

    def IPCA(x,test_size):
        clf_ipca = IncrementalPCA(n_components=4, batch_size=20)
        #x = np.concatenate((x_train,x_test),axis=0)
        x = clf_ipca.fit_transform(x)
        x_train,x_test= train_test_split(x,test_size=test_size,shuffle=False)
        #model_name = 'IPCA'
        return x_train,x_test
    def KNN(x_train,y_train,x_test):  
        model_classification =  KNeighborsClassifier()
        model_classification.fit(x_train, y_train)
        score = model_classification.score(x_train,y_train)
        print('score is: ',score)
        y_predict = model_classification.predict(x_test)
        model_name = 'KNN'
        return y_predict, model_name
    def NC(x_train,y_train,x_test):  
        model_classification =  NearestCentroid()
        model_classification.fit(x_train, y_train)
        score = model_classification.score(x_train,y_train)
        print('score is: ',score)
        y_predict = model_classification.predict(x_test)
        model_name = 'NC'
        return y_predict, model_name
    def DBSCANClustering(x_train,y_train,x_test,test_size):  
        x = np.concatenate((x_train,x_test),axis=0)
        scaler=MinMaxScaler()
        x = scaler.fit_transform(x)
        #model_rbm = BernoulliRBM(random_state=0, verbose=True, learning_rate=0.08, n_iter=100, n_components=2000)
        #x = model_rbm.fit_transform(x)
        
        #x_test = model_rbm.fit_transform(x_test)
        x_train, x_test = myClustering.IPCA(x,test_size)


        kde=KernelDensity(kernel='gaussian').fit(x_train)
        dens=kde.score_samples(x_test)
        #print(dens)            

        average_dens=np.mean(dens)
        if abs(average_dens)>=200:
            min_samples_=2
        else:
            min_samples_=int(abs(average_dens))
        print('avergae density is: ', average_dens)  
        
        x_train_ = np.array(x_train)
        dist=[]
        for i in range (0,len(x_train)):
            #print(x_train_ipca[i])
            dist_=cdist(np.array(np.reshape(x_train[i],(1,len(x_train[i])))),x_train_,'euclidean')
            average_dist_ = np.mean(dist_)
            dist.append(average_dist_)
        #print(dist)
        average_dist=max(dist)

        print('avergae distance is: ', average_dist)
            

        
        eps_=average_dist/(2*min_samples_)
        '''

        tree=KDTree(x_train)

        dist,ind = tree.query(x_train,k=int(min_samples_))

        
        #scaler=MinMaxScaler()
        #scaler.fit(dist)
        #dist=scaler.transform(dist)
        dist_=[]
        for i in range (len(dist)):
            print(dist[i])
            if max(dist[i])!=0:
                dist_.append(max(dist[i]))
            i+=1
        
        eps_=np.mean(dist_)/2
        '''
        print("eps is: ", eps_)
        print("min_samples are: ", min_samples_)
        model_classification = DBSCAN(eps=eps_,min_samples=min_samples_)
        model= model_classification.fit(x_train,y_train)
        
        #score = model_classification.score(x_train,y_train)
        #print('score is: ',score)
        y_predict = model_classification.fit_predict(x_test)
        y_predict = [0 if v == -1 else v for v in y_predict]
        print(y_predict)
        #y_predict_train, y_predict = train_test_split(y_predict_,test_size=0.2,shuffle=False)
        model_name = 'DBSCAN'
        return y_predict,model_name




