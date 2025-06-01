import tf_keras as keras
from keras.models import Sequential
from keras.layers import Embedding, Dense, Activation, Dropout, Flatten, BatchNormalization,Conv1D, MaxPooling1D, ELU, PReLU, LeakyReLU
from keras.layers import UpSampling1D, MaxPooling1D,AveragePooling1D
from keras.layers import LSTM
from keras import losses
from keras import optimizers
from keras import utils

import tensorflow as tf
import pandas as pd
import numpy as np
import struct
import sys
import datetime
from datetime import timedelta
import os
import pickle
import math
import random
import sklearn
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize, scale
import datetime
from datetime import timedelta

class myDCNN():
    def DCNN(input,num_classes):
        
        #vocab_size = len(input)
        model = Sequential()
        #print(input.shape[0],input.shape[1])
        #model.add(Embedding(vocab_size ,16))
        model.add(Dense(128,input_shape=(input.shape[1],input.shape[2])))
        print(model.output_shape)
        model.add(Conv1D(filters = 128, kernel_size =2,padding='same',kernel_initializer='uniform'))
        print(model.output_shape)
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(0.2))
        model.add(MaxPooling1D(pool_size=2, padding='same'))
        model.add(Dropout(0.1))
        model.add(Conv1D(filters = 64, kernel_size =2,padding='same',kernel_initializer='uniform'))
        print(model.output_shape)
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(0.2))
        model.add(MaxPooling1D(pool_size=2, padding='same'))
        model.add(Dropout(0.1))
        print(model.output_shape)
        model.add(Conv1D(filters = 32, kernel_size =2,padding='valid',kernel_initializer='uniform'))
        print(model.output_shape)
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(0.2))
        model.add(MaxPooling1D(pool_size=1, padding='same'))
        model.add(Dropout(0.1))
        #model.add(Conv1D(filters = 1, kernel_size =2,padding='valid',kernel_initializer='uniform'))
        #print(model.output_shape)
        #model.add(BatchNormalization(momentum=0.8))
        #model.add(LeakyReLU(0.2))
        #model.add(MaxPooling1D(pool_size=3, padding='same'))
        #model.add(Dropout(0.1))
        model.add(Conv1D(filters = 1, kernel_size =1,padding='valid',activation='relu',kernel_initializer='uniform'))
        print(model.output_shape)
        #model.add(BatchNormalization(momentum=0.8))
        #model.add(LeakyReLU(0.2))
        #model.add(MaxPooling1D(pool_size=3, padding='same'))
        model.add(Dropout(0.2))
        print(model.output_shape)
        #model.add(AveragePooling1D(pool_size=1,padding='valid'))
        model.add(Flatten())
        print(model.output_shape)
        model.add(Dense(200,activation='relu',kernel_initializer='uniform'))
        #model.add(Dense(90,kernel_initializer='uniform'))
        #model.add(LeakyReLU(alpha=0.2))
        #model.add(Dropout(0.1))
        #model.add(keras.layers.core.Reshape([input.shape[2],input.shape[1]]))
        model.add(Dense(num_classes,activation='softmax',kernel_initializer='uniform'))
        #model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['categorical_accuracy'])
        model.compile(loss='mean_squared_error',optimizer=optimizer,metrics=['mae'])
        return model

    def train(x_train,y_train,x_test,num_classes,test_size):
        global optimizer

        optimizer = optimizers.Adamax(learning_rate=0.008)
        x_train = np.array(x_train)
        x_test = np.array(x_test)

        x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
        x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
        #print(len(df_new_))

        #print(x_train.shape[0])
        print(num_classes)
        y_train = np.reshape(y_train,(len(y_train),1))
        y_train=utils.to_categorical(y_train,num_classes)
        #print(y_train)
        model = myDCNN.DCNN(x_train,num_classes)
        #print(np.count_nonzero(x_train))
        #model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['categorical_accuracy'])
        model.compile(loss='mean_squared_error',optimizer=optimizer,metrics=['mae'])
        history_callback = model.fit(x_train,y_train,batch_size=128,epochs=200,validation_split=test_size,verbose=2)

        print(model.summary())

        history_dict = history_callback.history
        loss_values = history_dict['loss']
        val_loss_values = history_dict['val_loss']

        mae_values = history_dict['mae']
        val_mae_values = history_dict['val_mae']
        #acc_values = history_dict['categorical_accuracy']
        #val_acc_values = history_dict['categorical_accuracy']
        df_val = pd.DataFrame()  
        df_val['loss_values'] = loss_values 
        df_val['val_loss_values'] = val_loss_values
        df_val['mae_values'] = mae_values
        df_val['val_mae_values'] = val_mae_values
        #df_val['acc_values'] = acc_values 
        #df_val['val_acc_values'] = val_acc_values 
        df_val.to_csv('validation_result.csv')


        y_train_predict=model.predict(x_train) 
        print(y_train_predict)
        trainScore=math.sqrt(mean_squared_error(y_train_predict,y_train)) 
        print('Train Score: %.5f RMSE' % (trainScore))
        
        
        y_predict_=model.predict(x_test)
        #print(x_predict)
        #print(currentTime)
        #print(y_predict_)
        y_predict=[]

        for i in range(len(y_predict_)):
            print(len(y_predict_[i]))
            y_predict.append(np.argmax(y_predict_[i]))
        '''
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
                        y_predict.append(j)'
        '''
        model_name = 'DCNN'
        return y_predict, model_name