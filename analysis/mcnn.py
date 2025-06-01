import pandas as pd
import numpy as np

import tf_keras
from tf_keras.models import Sequential, Model
from tf_keras.layers import Embedding, Dense, Activation, Dropout, Flatten, BatchNormalization,Conv1D, MaxPooling1D, ELU, PReLU, LeakyReLU
from tf_keras.layers import UpSampling2D, Conv2D,MaxPooling2D,AveragePooling1D,AveragePooling2D
from tf_keras.layers import LSTM
from tf_keras import losses
from tf_keras.optimizers import Adamax, SGD, Adam
from tf_keras import utils
import tensorflow as tf

import struct
import sys
import datetime
from datetime import timedelta
import os
import pickle
import matplotlib.pyplot as plt
import math
import random
import sklearn
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize, scale
import matplotlib.dates as mdates
import datetime
from datetime import timedelta
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score, roc_auc_score,roc_curve, precision_recall_curve,auc, f1_score,silhouette_score,normalized_mutual_info_score, adjusted_rand_score
#import similarity
#from similarity import similarityAlgo
#os.environ['TF_USE_LEGACY_tf_keras']='1'
class myMCNN(tf_keras.Model):

    def __init__(self,vocab_size,num_classes):
        super().__init__()
        #self.layer_embedding = Embedding(vocab_size,128)
        self.layer_dense_0 = Dense(units=128, activation='relu')
        self.layer_conv_1 = Conv1D(filters=64,kernel_size=1,padding='same',activation='relu')
        self.layer_conv_2 = Conv1D(filters=64,kernel_size=2,padding='same',activation='relu')
        self.layer_conv_3 = Conv1D(filters=64,kernel_size=3,padding='same',activation='relu')
        self.layer_maxpool = MaxPooling1D(pool_size=2)
        self.layer_dropout = Dropout(rate=0.2)
        self.layer_flatten = Flatten()
        self.layer_dense_1 = Dense(units=128,activation='relu')
        self.layer_dense_2 = Dense(units=num_classes,activation='relu')
    def call(self,inputs):
        #x_0=self.layer_embedding(inputs)
        x_0 = self.layer_dense_0(inputs)
        print(x_0.shape)
        x_1 = self.layer_conv_1(x_0)
        x_1 = self.layer_maxpool(x_1)
        print(x_1.shape)
        x_2 = self.layer_conv_2(x_0)
        x_2 = self.layer_maxpool(x_2)
        print(x_2.shape)
        x_3 = self.layer_conv_3(x_0)
        x_3 = self.layer_maxpool(x_3)
        print(x_3.shape)
        x = tf.concat([x_1,x_2,x_3],axis=-1)
        x = self.layer_flatten(x)
        x = self.layer_dense_1(x)
        x = self.layer_dropout(x)
        output = self.layer_dense_2(x)
        print('the shape of output is: ',output.shape)
        return output
    def train(x_train,y_train,x_test, num_classes,test_size):

        x_train = np.array(x_train)
        x_test = np.array(x_test)
        print(x_train)
        #emb_dim_=200
        #filters_=100
        #units_=256
        #dropout_rate_=0.2
        vocab_size = len(x_train)
        #print(x_train.shape[0])

        x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
        x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
        y_train = np.reshape(y_train,(len(y_train),1))
        y_train=utils.to_categorical(y_train,num_classes)
        model = myMCNN(vocab_size=vocab_size,num_classes=num_classes)
        #print(type(myMCNN()))      
        #print(y_train)
        optimizer = Adamax(learning_rate=0.001)
        model.compile(loss='mean_squared_error',optimizer=optimizer,metrics=['mae'])

        history_callback = model.fit(x_train,y_train,batch_size=128,epochs=200,validation_split=test_size,verbose=2)
        
        history_dict = history_callback.history
        loss_values = history_dict['loss']
        val_loss_values = history_dict['val_loss']

        mae_values = history_dict['mae']
        val_mae_values = history_dict['val_mae']
        df_val = pd.DataFrame()  
        df_val['loss_values'] = loss_values 
        df_val['val_loss_values'] = val_loss_values
        df_val['mae_values'] = mae_values
        df_val['val_mae_values'] = val_mae_values
        df_val.to_csv('validation_result.csv')

        y_train_predict=model.predict(x_train) 
        #print(y_train_predict)
        trainScore=math.sqrt(mean_squared_error(y_train_predict,y_train)) 
        print('Train Score: %.5f RMSE' % (trainScore))
        y_predict_ = model.predict(x_test)
        y_predict=[]
        for i in range(len(y_predict_)):
            y_predict.append(np.argmax(y_predict_[i]))
        print(y_predict)
        '''
        y_predict=[]

        for i in range(len(y_predict_)):
            max_value = max(y_predict_[i])
            #print(max_value)
            if max_value == 0:
                #y_predict.append(4)
                relationships = x_test_orig[i]
                #print(relationships)                
                selected_classification = similarityAlgo.similarity(relationships,classes)
                print(i,selected_classification)
                y_predict.append(selected_classification )  
            else:
                for j in range(len(y_predict_[i])):
                    if y_predict_[i][j] == max_value:
                        print(i,j,max_value)
                        y_predict.append(j)
            '''
        model_name = 'MCNN'
        return y_predict, model_name



