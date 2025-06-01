import torch
import pandas as pd
import numpy as np
import random
import nltk
from nltk import word_tokenize
import os
import json
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score, roc_auc_score,roc_curve, precision_recall_curve,auc, f1_score,silhouette_score,normalized_mutual_info_score, adjusted_rand_score
import datetime
import time
import utils
from utils import rel_utils,tokenizer
from utils.rel_utils import preprocess
from utils.tokenizer import MLTokenizer
import re
import gc
import analysis
from analysis import cnn,dcnn,deepae,dcgan,svm,clustering,nn_sklearn,mcnn,regression,tree,selftraining,naivebayes
from analysis.nn_sklearn import myNN
from analysis.mcnn import myMCNN
from analysis.regression import myRegression
from analysis.selftraining import selfTrainingClassifier
from analysis.tree import myTree
from analysis.naivebayes import myNaiveBayes
from analysis.dcnn import myDCNN
from analysis.dcgan import myDCGAN
from analysis.deepae import myAutoEncoder
from analysis.cnn import myCNN
from analysis.clustering import myClustering
from analysis.svm import mySVM


#from imblearn.metrics import classification_report_imbalanced
#from imblearn.pipeline import make_pipeline as make_pipeline_imb
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()
torch.cuda.empty_cache()
nltk.download('punkt')

'''
class benchmarkSpacy

class benchmarkBert

class benchmarkFastText

'''
class textClassifications():
    def dataPreprocessing(df,class_dir):
        
        global classes
        


        x_orig =[]
        for i in range(len(df)):
            string = df['LLM Thought'][i].split('{')[2].split('}')[0]
            #print(string)
            #x_orig.append(string)
            
            
            c = re.findall('\n',string)
            #print(len(c))
            input_array=[]
            for j in range(len(c)):
                input_array.append(string.split('\n')[j])
            #print(input_array)
            x_orig.append(input_array)
            

        #print(x_orig)
        y_orig = df['Alignment']
        #print(len(x_orig[0]))
        #print(len(classes))
        classifications,classes = preprocess.loadClasses(class_dir)
        #print(len(classes))
        '''
        classes_tokenized = MLTokenizer.tokenizer(classes).argmax(axis=1)
        classifications['classes_tokenized'] = classes_tokenized
        classifications.to_csv("classes_tokenized.csv")
        '''
        y = []

        for i in range(len(df)):
            for j in range(len(classifications)):
                if df['Alignment'][i] == classifications['classes'][j]:
                    #print(i,df['classes'][i], classifications['classes_en'][j])
                    y.append(classifications['index'][j])
        return x_orig, y_orig, y,classifications,classes

    def NBAnalysis(x_orig, y_orig, y, classifications):
        global num_classes
        num_classes = len(set(y))
        print(num_classes)
        print(set(y))
        #x_train_orig,x_test_orig,y_train_orig,y_test_orig = train_test_split(x_orig,y_orig,test_size=0.1,shuffle=False)
     
        #x =  MLTokenizer.word2vec(x_orig)
        #x =  MLTokenizer.text2vec(x_orig)
        #x =  MLTokenizer.tokenizerTFID(x_orig)  
        #x =  MLTokenizer.tokenizerCount(x_orig)
        #x =  MLTokenizer.tokenizerHashing(x_orig)
        x =  MLTokenizer.tokenizerHybrid(x_orig)

        #print(y)
        #print(len(x))
        #print(len(y))
        x_train_orig,x_test_orig,y_train_orig,y_test_orig ,x_train,x_test,y_train,y_test= train_test_split(x_orig,y_orig,x,y,test_size=test_size)
        #x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=test_size,shuffle=False)
        y_predict_tokenized,model_name = mySVM.LinearSVC(x_train,y_train,x_test)
        #y_predict_tokenized,model_name = myClustering.KNN(x_train,y_train,x_test)
        #y_predict_tokenized,model_name = myClustering.NC(x_train,y_train,x_test)
        #y_predict_tokenized,model_name = myClustering.DBSCANClustering(x_train,y_train,x_test,test_size)
        #y_predict_tokenized,model_name = myCNN.train(x_train,y_train,x_test,x_test_orig, num_classes,classes,test_size)
        #y_predict_tokenized,model_name = myMCNN.train(x_train,y_train,x_test,x_test_orig, num_classes,classes,test_size)
        #y_predict_tokenized,model_name = myAutoEncoder.train(x_train,y_train,x_test,y_test,x_test_orig,num_classes)
        #y_predict_tokenized,model_name = myDCNN.train(x_train,y_train,x_test,x_test_orig, num_classes,classes,test_size)
        #y_predict_tokenized,model_name = myRegression.RidgeClassifier(x_train, y_train,x_test)
        #y_predict_tokenized,model_name = myRegression.SGDClassifier(x_train, y_train,x_test)
        #y_predict_tokenized,model_name = myRegression.LogisticRegression(x_train,y_train,x_test)
        #y_predict_tokenized,model_name = myNN.BernoulliRBM(x_train,y_train,x_test,y_test,test_size)
        #y_predict_tokenized,model_name = myNN.MLP(x_train,y_train,x_test)
        #y_predict_tokenized,model_name = myTree.DecisionTree(x_train,y_train,x_test)
        #y_predict_tokenized,model_name = myTree.RandomForest(x_train,y_train,x_test)
        #y_predict_tokenized,model_name = myNaiveBayes.BernoulliNB(x_train,y_train,x_test)
        #y_predict_tokenized,model_name = myNaiveBayes.categoricalNB(x_train,y_train,x_test)
        #y_predict_tokenized,model_name = myNaiveBayes.ComplementNB(x_train,y_train,x_test)
        #y_predict_tokenized,model_name = myNaiveBayes.GaussianNB(x_train,y_train,x_test)
        #y_predict_tokenized,model_name = myNaiveBayes.multinomialNB(x_train,y_train,x_test)
        #y_predict_tokenized,model_name = selfTrainingClassifier.selfTrainingClassifier(x_train, y_train,x_test)
        #y_predict_tokenized,model_name = selfTrainingClassifier.LabelSpreading(x_train, y_train,x_test)
        #y_predict_tokenized,model_name = myDCGAN.train(x_train,y_train,x_test,y_test,x_test_orig,num_classes)
        #print(y_predict_tokenized )
        
        y_predict=[]
        for i in range(len(y_predict_tokenized )):
            for j in range(len(classifications)):
                #print(y_predict_tokenized[i])
                if y_predict_tokenized[i] == classifications['index'][j]:
                    y_predict.append(classifications['classes'][j])
        #print(y_predict)
        return x_train_orig,x_test_orig,y_train_orig,y_test_orig, y_predict, y_test,y_predict_tokenized,model_name
    def generateOutput(x_test_orig,y_test_orig,y_test,y_predict,y_predict_tokenized):
        y_actual=[]
        for i in range(len(y_test)):
            for j in range(len(classifications)):
                #print(max(y_test[i]))
                if y_test[i] == classifications['index'][j]:
                    y_actual.append(classifications['classes'][j])
        #begin = -int(np.ceil((0.1*(len(x_orig)))))
        print(len(x_test_orig))
        df_test=pd.DataFrame()
        df_test['Alignment_Factor'] = x_test_orig
        df_test['Alignment'] = y_actual
        df_test['predicted_Alignment'] = y_predict
        df_test = df_test.reset_index()
        #print(df_test)

        accuracy=0
        for i in range(len(df_test)):
            if df_test['Alignment'][i]==df_test['predicted_Alignment'][i]:
                accuracy=accuracy+1
        print('accuracy is: ',accuracy/len(df_test))
        df_test.to_csv(output_dir)
        accuracy_score = f1_score(y_test, y_predict_tokenized, average="micro")
        print('f1 score is: ',accuracy_score)

        return df_test, accuracy,accuracy_score
current_time = str(datetime.datetime.now())[:18]
current_time = current_time.replace(':','')
#data_dir = '/userhome/35/xtan/KG_Classification/original_data.csv'
project_dir = '/Users/sharontan6217/Documents/KG_Classification/'
os.chdir(project_dir)
print(os.getcwd())
data_dir = './climate_scenarios_regression_data_deepseek.xlsx'
class_dir = './classes.csv'

test_size=0.1
if __name__=='__main__':

    df_orig = pd.read_excel(data_dir)
    
    df_orig = df_orig.drop('index',axis=1)
    print(df_orig.columns)
    llm_models = set(df_orig['LLM Model'])
    print(llm_models)
    for i in range(10):
        for llm_model in llm_models:
            gc.collect()
            df = df_orig[df_orig['LLM Model']==llm_model]
            df = df.reset_index()

            x_orig, y_orig, y, classifications, classes = textClassifications.dataPreprocessing(df,class_dir)
            
            #x_train_orig,x_test_orig,y_train_orig,y_test_orig = train_test_split(x_orig,y_orig,test_size=test_size,shuffle=False)
            #print(y_test_orig)
            x_train_orig,x_test_orig,y_train_orig,y_test_orig, y_predict, y_test, y_predict_tokenized,model_name = textClassifications.NBAnalysis(x_orig, y_orig, y, classifications)
            if os.path.exists('./output/'+model_name+'/')==False:
                os.mkdir('./output/'+model_name+'/')
            output_dir = './output/'+model_name+'/result_'+llm_model+'_'+current_time+'.csv'

            df_test, accuracy,accuracy_score = textClassifications.generateOutput(x_test_orig,y_test_orig,y_test,y_predict, y_predict_tokenized)
            with open ('./output/'+model_name+'/result_'+model_name+'.log','a') as f:
                f.write('\n')
                #f.write(current_time+','+str(accuracy)+','+str(accuracy_score))
                f.write(llm_model+','+model_name+','+current_time+','+str(accuracy)+','+str(accuracy_score))
                f.close()
            #if model_name in ['CNN','MCNN','AutoEncoder','DCNN']:
                #time.sleep(10)