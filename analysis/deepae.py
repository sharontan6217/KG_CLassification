import pandas as pd
import numpy as np
import struct
import sys
import inspect
import sklearn
from sklearn import cluster, preprocessing
from sklearn.preprocessing import normalize, scale, MinMaxScaler, LabelBinarizer
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score, roc_auc_score,roc_curve, precision_recall_curve,auc, f1_score,silhouette_score,normalized_mutual_info_score, adjusted_rand_score
from sklearn.model_selection import train_test_split 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA, KernelPCA, IncrementalPCA, FastICA
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import torch
from torch import reshape,cat,multiply, flatten
from torch.nn import init,Sequential, Module
from torch.nn import Linear, BatchNorm1d,BatchNorm2d,BatchNorm3d, Dropout, Dropout2d, Flatten, ZeroPad2d, ConvTranspose2d, ConvTranspose1d
from torch.nn import LeakyReLU, ReLU, Softmax
from torch.nn import UpsamplingBilinear2d, Upsample,Conv2d,MaxPool2d,AvgPool1d,AvgPool2d,Conv1d,AdaptiveAvgPool1d,AdaptiveMaxPool1d, MaxPool1d,UpsamplingNearest2d,MaxPool3d
from torch.nn import MSELoss,L1Loss,CrossEntropyLoss,SoftMarginLoss,BCELoss,BCEWithLogitsLoss,HingeEmbeddingLoss
from torch.autograd import Variable
from torch.optim import Adamax, Adam
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple

import tensorflow as tf

#from torch.nn import utils
from tf_keras import utils

from tensorflow.python.client import device_lib

import scipy
from scipy.io import wavfile
from scipy.signal import butter,lfilter,filtfilt,lfilter_zi,sosfilt
import pickle
import random
from scipy.spatial.distance import sqeuclidean
import datetime
from datetime import timedelta

import os
import importlib
import math

learning_rate=2e-3
criterion=torch.nn.MSELoss()

batch_size=128
iterations=200

evaluation_rate=0.015


#tf.compat.v1.disable_eager_execution()
#device=torch.device("mps")
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype=torch.int64


    
class myAutoEncoder(torch.nn.Module):
    
    def __init__(self,emb_dim,num_classes):
        self.emb_dim = emb_dim
        self.num_classes = num_classes
              
        super(myAutoEncoder,self).__init__()
        
        self.encoderLayer_0=Sequential(torch.nn.Linear(in_features=1,out_features=emb_dim))
       
        self.encoderLayer_1=Sequential(Conv2d(1,16,kernel_size=2,stride=1,padding=1),
                                       BatchNorm2d(num_features=16,momentum=0.8),                                       
                                       LeakyReLU(0.2),
                                       MaxPool2d(2,stride=2),
                                       Dropout2d(0.1))
        self.encoderLayer_2=Sequential(Conv2d(16,32,kernel_size=2,stride=1,padding=1),
                                       BatchNorm2d(num_features=32,momentum=0.8),
                                       LeakyReLU(0.2),
                                       MaxPool2d(2,stride=2),
                                       Dropout2d(0.1))

        
        self.encoderLayer_3=Sequential(Conv2d(32,64,kernel_size=2,stride=1,padding=1),
                                       BatchNorm2d(num_features=64,momentum=0.8),
                                       LeakyReLU(0.2),
                                       MaxPool2d(3,stride=2),
                                       Dropout2d(0.1))
        '''
        self.encoderLayer_4=Sequential(Conv2d(64,128,kernel_size=2,stride=1,padding=1),
                                       BatchNorm2d(num_features=128,momentum=0.8),
                                       LeakyReLU(0.2),
                                       MaxPool2d(3,stride=3),
                                       Dropout2d(0.1))
        
        self.encoderLayer_5=Sequential(Conv2d(256,512,kernel_size=2,stride=1,padding=1),
                                       BatchNorm2d(num_features=512,momentum=0.8),
                                       LeakyReLU(0.2),
                                       MaxPool2d(7,stride=2),
                                       Dropout2d(0.1))
        self.decoderLayer_1=Sequential(ConvTranspose2d(512,256,kernel_size=1),
                                       BatchNorm2d(num_features=256,momentum=0.8),
                                       LeakyReLU(0.2),
                                       UpsamplingNearest2d(scale_factor=2),
                                       Dropout2d(0.1))
        
        self.decoderLayer_1=Sequential(ConvTranspose2d(128,64,kernel_size=1),
                                       BatchNorm2d(num_features=64,momentum=0.8),
                                       LeakyReLU(0.2),
                                       UpsamplingNearest2d(scale_factor=3),
                                       Dropout2d(0.1))
        '''
        self.decoderLayer_1=Sequential(ConvTranspose2d(64,32,kernel_size=1),
                                       BatchNorm2d(num_features=32,momentum=0.8),
                                       LeakyReLU(0.2),
                                       UpsamplingNearest2d(scale_factor=2),
                                       Dropout2d(0.1))
        
        self.decoderLayer_2=Sequential(ConvTranspose2d(32,16,kernel_size=1),
                                       BatchNorm2d(num_features=16,momentum=0.8),
                                       LeakyReLU(0.2),
                                       UpsamplingNearest2d(scale_factor=2),
                                       Dropout2d(0.1))
        self.decoderLayer_3=Sequential(ConvTranspose2d(16,1,kernel_size=1),
                                       BatchNorm2d(num_features=1,momentum=0.8),
                                       LeakyReLU(0.2),
                                       UpsamplingNearest2d(scale_factor=2),
                                       Dropout2d(0.1))   

        self.fc_1=Linear(emb_dim,128,bias=True)
        torch.nn.init.xavier_uniform_(self.fc_1.weight)
        self.fcLayer_1=Sequential(self.fc_1,
                                ReLU(),
                                Dropout2d(0.2))
        self.fc_2=Linear(128,num_classes,bias=True)
        torch.nn.init.xavier_uniform_(self.fc_2.weight)
        self.fcLayer_2=Sequential(self.fc_2,Softmax(2))
        
        
        
        

    
    

    
    def autoEncoder(self,X):
        #print(X.shape)
        encoder=self.encoderLayer_0(X)
        #print("encoder_0's shape: ",encoder.shape)
        encoder=self.encoderLayer_1(encoder)
        #print("encoder_1's shape: ",encoder.shape)
        encoder=self.encoderLayer_2(encoder)
        #print("encoder_2's shape: ",encoder.shape)

        encoder=self.encoderLayer_3(encoder) 
        #print("encoder_3's shape: ",encoder.shape)
        '''
        encoder=self.encoderLayer_4(encoder)
        print("encoder_4's shape: ",encoder.shape)
        
        encoder=self.encoderLayer_5(encoder)
        print("encoder_5's shape: ",encoder.shape)
        '''

        decoder=self.decoderLayer_1(encoder)
        #print("decoder_1's shape: ",decoder.shape)
        decoder=self.decoderLayer_2(decoder)
        #print("decoder_2's shape: ",decoder.shape)
        decoder=self.decoderLayer_3(decoder) 
        #print("decoder_3's shape: ",decoder.shape)
        '''
        decoder=self.decoderLayer_4(decoder)
        print("decoder_4's shape: ",decoder.shape)
        
        decoder=self.decoderLayer_5(decoder)
        print("decoder_5's shape: ",decoder.shape)
        '''

        fc1=self.fcLayer_1(decoder)
        #print(fc1.shape)
        fl=flatten(fc1,start_dim=1,end_dim=2)
        #print(fl.shape)
        out=self.fcLayer_2(fl)
        #print(out.shape)
        #out=reshape(fc2,(fc2.shape[0],fc2.shape[2],fc2.shape[1]))
        
        
        #d_loss=self.decoder.train_on_batch(encoded,decoded)
        return out

    
    def train(x_train,y_train,x_test,y_test, num_classes):
        num_classes = num_classes+1
        x_train = np.array(x_train)
        x_test= np.array(x_test)
        #y_train = np.array(y_train)
        #y_test= np.array(y_test)
        x_train = np.reshape(x_train,(x_train.shape[0],1,x_train.shape[1],1))
        x_test = np.reshape(x_test,(x_test.shape[0],1,x_test.shape[1],1))
        #print(len(df_new_))

        #print(x_train.shape[0])
        y_train = np.reshape(y_train,(len(y_train),1,1))
        #print(num_classes)
        
        y_train=utils.to_categorical(y_train,num_classes)
        #print(y_train.shape)

        y_test = np.reshape(y_test,(len(y_test),1,1))
        y_test =utils.to_categorical(y_test,num_classes)
   
        dataset_train=TensorDataset(torch.Tensor(x_train),torch.Tensor(y_train))
           
        dataset_test=TensorDataset(torch.Tensor(x_test),torch.Tensor(y_test))  

        data_train=torch.utils.data.DataLoader(dataset=dataset_train,batch_size=batch_size,shuffle=True)
        data_test=torch.utils.data.DataLoader(dataset=dataset_test,batch_size=batch_size,shuffle=True)
        #emb_dim = x_train.shape[2]
        emb_dim=128
        model=myAutoEncoder(emb_dim,num_classes)
        '''
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
        '''
        model=model.to(device)
        
        optimizer=Adamax(params=model.parameters(),lr=learning_rate)

     

        
        train_cost_=[]
        

        for epoch in range (iterations):

            avg_cost_train=0
            train_cost=[]
            train_accuracy=[]

            avg_cost_train_=[]
            y_train_predict=[]
            for i, (batch_X,batch_Y) in enumerate (data_train):
                X=Variable(batch_X)
                Y=Variable(batch_Y)
                
                X,Y=X.to(device),Y.to(device)
                #print(i)
                


                
                optimizer.zero_grad()
                
                y_train_predict_=model.autoEncoder(X)
                y_train_predict_ = y_train_predict_[:,0:1,:]
                Y_=Y.data.max(dim=0)[0]               
                #Y=reshape(Y,(Y.shape[0],Y.shape[1]))
                #print('shape of train_predict:',train_predict.shape)
                #print('shape of Y:',Y.shape)
                cost=criterion(y_train_predict_,Y)
                
                cost.backward()
                optimizer.step()

                #y_train_predict_= reshape(y_train_predict_,(y_train_predict_.shape[0],y_train_predict_.shape[2],y_train_predict_.shape[1]))           
                train_prediction=y_train_predict_.data.max(dim=0)[0]

                #print(train_prediction.data)

                train_accuracy.append((((train_prediction.data-Y_.data)<evaluation_rate).float().mean()).item())
                train_cost.append(cost.item())
                train_cost_.append(cost.item())
                avg_cost_train_.append(max(train_cost))
                y_train_predict.append(y_train_predict_.detach().cpu().numpy())

                if i % 1==0:
                    print("Epoch={},\t batch={},\t cost={:2.4f},\t accuracy={}".format(epoch+1,i,train_cost[-1],train_accuracy[-1]))

                i+=1
 
            if epoch==0:
                print("epoch is:", epoch+1)
                avg_cost_train += max(train_cost)
            else:
                avg_cost_train +=min(avg_cost_train_)

                 
            print("[Epoch:{:>4}], averaged cost={:>.9}".format(epoch+1,avg_cost_train))
            #train_cost_.append(avg_cost_train)
            del train_cost
                
            epoch+=1
        print(y_train_predict)
        print(len(train_cost_))
        y_train_predict=np.concatenate(y_train_predict,axis=0)




        print(y_train_predict.shape)
        print(y_train.shape)
        
        #model.eval()
        y_test_predict=[]
        for j, (batch_x_test,batch_y_test) in enumerate (data_test):
            X_=Variable(batch_x_test)

          
            X_=X_.to(device)
          
            optimizer.zero_grad()

            y_predict_=model.autoEncoder(X_)
            y_predict_ = y_predict_[:,0:1,:]
            #y_predict_= reshape(y_predict_,(y_predict_.shape[0],y_predict_.shape[2],y_predict_.shape[1]))
            y_test_predict.append(y_predict_.detach().cpu().numpy())
            j+=1
        y_test_predict=np.concatenate(y_test_predict,axis=0)
        print(y_test_predict.shape)
        print(y_test.shape)
        y_train_predict=np.reshape(y_train_predict,(int(y_train_predict.shape[0]*y_train_predict.shape[1]),y_train_predict.shape[2]))
        y_train=np.reshape(y_train,(int(y_train.shape[0]*y_train.shape[1]),y_train.shape[2]))
        y_test_predict=np.reshape(y_test_predict,(int(y_test_predict.shape[0]*y_test_predict.shape[1]),y_test_predict.shape[2]))
        y_test=np.reshape(y_test,(int(y_test.shape[0]*y_test.shape[1]),y_test.shape[2]))
        y_predict=[]
        for i in range(len(y_test_predict)):
            print(len(y_test_predict[i]))
            y_predict.append(np.argmax(y_test_predict[i]))
        '''
        for i in range(len(y_test_predict)):
            max_value = max(y_test_predict[i])
            #print(max_value)
            if max_value == 0:
                #y_predict.append(4)
                relationships = x_test_orig[i]
                #print(relationships)                
                selected_classification = similarityAlgo.similarity(relationships,classes)
                print(i,selected_classification)
                y_predict.append(selected_classification )  
            else:
                for j in range(len(y_test_predict[i])):
                    if y_test_predict[i][j] == max_value:
                        print(i,j,max_value)
                        y_predict.append(j)


        '''
        #testScore=math.sqrt(mean_squared_error(y_test,y_predict)) 
        #print('Test Score: %.5f RMSE' % (testScore))
        model_name = 'AutoEncoder'
        return y_predict, model_name