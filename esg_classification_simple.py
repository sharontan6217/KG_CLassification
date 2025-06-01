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

    def NBAnalysis(x, y):
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
        #x =  MLTokenizer.tokenizerHybrid(x_orig)

        #print(y)
        #print(len(x))
        #print(len(y))
        x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=test_size)
        #x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=test_size,shuffle=False)
        #y_predict,model_name = mySVM.LinearSVC(x_train,y_train,x_test)
        #y_predict,model_name = myClustering.KNN(x_train,y_train,x_test)
        #y_predict,model_name = myClustering.NC(x_train,y_train,x_test)
        #y_predict,model_name = myClustering.DBSCANClustering(x_train,y_train,x_test,test_size)
        #y_predict,model_name = myCNN.train(x_train,y_train,x_test, num_classes,test_size)
        #y_predict,model_name = myMCNN.train(x_train,y_train,x_test,num_classes,test_size)
        #y_predict,model_name = myAutoEncoder.train(x_train,y_train,x_test,y_test,num_classes)
        #y_predict,model_name = myDCNN.train(x_train,y_train,x_test, num_classes,test_size)
        #y_predict,model_name = myRegression.RidgeClassifier(x_train, y_train,x_test)
        #y_predict,model_name = myRegression.SGDClassifier(x_train, y_train,x_test)
        #y_predict,model_name = myRegression.LogisticRegression(x_train,y_train,x_test)
        #y_predict,model_name = myNN.BernoulliRBM(x_train,y_train,x_test,y_test,test_size)
        #y_predict,model_name = myNN.MLP(x_train,y_train,x_test)
        #y_predict,model_name = myTree.DecisionTree(x_train,y_train,x_test)
        #y_predict,model_name = myTree.RandomForest(x_train,y_train,x_test)
        y_predict,model_name = myNaiveBayes.BernoulliNB(x_train,y_train,x_test)
        #y_predict,model_name = myNaiveBayes.categoricalNB(x_train,y_train,x_test)
        #y_predict,model_name = myNaiveBayes.ComplementNB(x_train,y_train,x_test)
        #y_predict,model_name = myNaiveBayes.GaussianNB(x_train,y_train,x_test)
        #y_predict,model_name = myNaiveBayes.multinomialNB(x_train,y_train,x_test)
        #y_predict,model_name = selfTrainingClassifier.selfTrainingClassifier(x_train, y_train,x_test)
        #y_predict,model_name = selfTrainingClassifier.LabelSpreading(x_train, y_train,x_test)
        #y_predict,model_name = myDCGAN.train(x_train,y_train,x_test,y_test,num_classes)
        #print(y_predict )
        
        return x_test,y_predict, y_test,model_name
    def generateOutput(x_test,y_test,y_predict):

        print(len(x_test))
        df_test=pd.DataFrame()
        df_test['Alignment_Factor'] = x_test
        df_test['Alignment'] = y_test
        df_test['predicted_Alignment'] = y_predict
        df_test = df_test.reset_index()
        #print(df_test)

        accuracy=0
        for i in range(len(df_test)):
            if df_test['Alignment'][i]==df_test['predicted_Alignment'][i]:
                accuracy=accuracy+1
        print('accuracy is: ',accuracy/len(df_test))
        df_test.to_csv(output_dir)
        accuracy_score = f1_score(y_test, y_predict, average="micro")
        print('f1 score is: ',accuracy_score)

        return df_test, accuracy,accuracy_score
current_time = str(datetime.datetime.now())[:18]
current_time = current_time.replace(':','')
#data_dir = '/userhome/35/xtan/KG_Classification/original_data.csv'
project_dir = '/userhome/35/xtan/KG_Classification/'
os.chdir(project_dir)
print(os.getcwd())
data_dir = './data/'
class_dir = './classes.csv'

test_size=0.25
if __name__=='__main__':
    
    for r,d,f in os.walk(data_dir):
        for f_ in f:
            print(f_)
            df_orig = pd.read_excel(r+f_)
            file_name = str(f_).lower()
            if 'deepseek' in file_name:
                llm_model='DeepSeek R1'
            elif 'anthropic' in file_name:
                llm_model='Anthropic Claude 3.5 Sonnet'       
            elif 'gpt' in file_name:
                llm_model='GPT-4o' 
            df_orig = df_orig.drop('index',axis=1) 
    
            x = []
            for i in range(len(df_orig)):
                ans0=df_orig['The climate-related scenarios and associated time horizon(s) considered'][i]
                ans1=df_orig['The potential impact of climate-related issues on financial performance (e.g., revenues, costs) and financial position (e.g., assets, liabilities)'][i]
                ans2=df_orig['How their strategies might change to address such potential risks and opportunities'][i]
                ans3=df_orig['Where they believe their strategies may be affected by climate-related risks and opportunities'][i]
                ans4=df_orig['Transparency in Strategy Testing and Outcomes'][i]
                ans5=df_orig['Strategic Resilience Testing vs. Impact Assessment'][i]
                ans6=df_orig['Preliminary or pilot scenario analysis'][i]
                ans7=df_orig['Comprehensive coverage of business areas and portfolio-specific risks'][i]
                ans8=df_orig['Exclusion of customer-specific transition plans'][i]
                ans9=df_orig['Progression from Qualitative to Quantitative Analysis'][i]
                ans10=df_orig['Sector-Specific Analysis Without Organization-Wide Application'][i]
                ans11=df_orig['Collaborative Scenario Development and Operational Risk Application'][i]
                x_ = np.array([ans0,ans1,ans2,ans3,ans4,ans5,ans6,ans7,ans8,ans9,ans10,ans11])
                x.append(x_)
            #print(x.dtype())
            print(len(x))
            y = df_orig['label'].values-1
            print(len(y))
            #print(y)

            for i in range(10):
                        
                #x_train_orig,x_test_orig,y_train_orig,y_test_orig = train_test_split(x_orig,y_orig,test_size=test_size,shuffle=False)
                #print(y_test_orig)
                x_test, y_predict, y_test,model_name= textClassifications.NBAnalysis(x,  y)
                if os.path.exists('./output_simple/'+model_name+'/')==False:
                    os.mkdir('./output_simple/'+model_name+'/')
                output_dir = './output_simple/'+model_name+'/result_'+llm_model+'_'+current_time+'.csv'

                df_test, accuracy,accuracy_score = textClassifications.generateOutput(x_test,y_test,y_predict)
                with open ('./output_simple/'+model_name+'/result_'+model_name+'.log','a') as f:
                    f.write('\n')
                    #f.write(current_time+','+str(accuracy)+','+str(accuracy_score))
                    f.write(llm_model+','+model_name+','+current_time+','+str(accuracy)+','+str(accuracy_score))
                    f.close()
