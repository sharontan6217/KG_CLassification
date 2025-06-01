import torch
import pandas as pd
import numpy as np
import random
import nltk
from nltk import word_tokenize
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score, roc_auc_score,roc_curve, precision_recall_curve,auc, f1_score,silhouette_score,normalized_mutual_info_score, adjusted_rand_score
import datetime
import time
import similarity
from similarity import similarityAlgo
import utils
from utils import rel_utils,tokenizer
from utils.rel_utils import preprocess
from utils.tokenizer import MLTokenizer
import analysis
from analysis import cnn,dcnn,deepae,dcgan,svm,clustering
from analysis.dcnn import myDCNN
from analysis.dcgan import myDCGAN
from analysis.deepae import myAutoEncoder
from analysis.cnn import myCNN
from analysis.clustering import myClustering
from analysis.svm import mySVM


#from imblearn.metrics import classification_report_imbalanced
#from imblearn.pipeline import make_pipeline as make_pipeline_imb
file_name = "Full_entities_table.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()
torch.cuda.empty_cache()
nltk.download('punkt')

'''
class benchmarkSpacy

class benchmarkBert

class benchmarkFastText

'''
class generateRelations():
    def textGenerateRelations(text):
        '''
        text = """
        The Los Angeles County Sheriff's Department said police were reported to Saugus High School in Santa Clarita to handle a shooting. It is alleged that an Asian male suspect dressed in black appeared at the scene of the crime. Police cordoned off the high school and other schools in the area and escorted the students away.Los Angeles County Sheriff Alex Villanueva said six people were shot in the incident, including the 16-year-old suspect. It was the suspect's birthday. The police did not disclose the suspect's motive for the attack. A video shows the gunman shooting himself in the head. Authorities allege that the gunman was armed with a .45 caliber pistol.A 16-year-old girl and a 14-year-old boy died. Three other wounded were hospitalized, and the gunman was seriously wounded.
        """
        '''
        kb = textClassification.from_text_to_kb(text)
        relations = kb.relations
        return relations
    def articleGenerateRelations(data_dir):
        orig = pd.read_csv(data_dir,header=1)

        print(orig.columns)
        df = orig[925:len(orig)]
        df = df.reset_index()
        relations=[]
        texts = df['新聞內容_en']
        for text in texts:
            kb = textClassification.from_text_to_kb(text)
            relations.append(kb.relations)
            subprocess.run(['sync'])

        df['relations'] =preprocess.featureExtraction(relations)

        df['classes']=df['文章類型_en']

        df = df[['relations','classes']]
        df =preprocess.featuresFiltering(df,num_features)
        #df['relations'] = df['features']
        #print(df)
        df.to_csv('/userhome/35/xtan/textClassification/df_relations_20_925_withFiltering.csv')
        return df
    def urlsGenerateRelations(urls):


        links, classes,timesequence,df_urls = collectURLs.loadLinks(keys=['Money Laundering'],pages=1,lang='en',region='HK',period='7d')
        kb = articlesClassification.from_urls_to_kb(links)
        kb.print()

        return kb
class textClassifications():
    def dataPreprocessing(relations_dir,class_dir):
        global classes
        df = pd.read_csv(relations_dir)

        print(df.columns)
        x_orig = []
        for i in range(len(df)):
            x_ = df['features'][i]
            #print(x_)

            relations = np.array(())
            count = x_.count(",")
            #print(count)

            for i in range(count):
              x = x_.split(',')[i]
              x = preprocess.dataClean(x)
              #x = word_tokenize(x)
              relations = np.append(relations,x)

            #x_orig.append(x_)
            x_orig.append(relations)
        #print(x_orig)
        #print(x_orig.shape)
        y_orig = df['classes']
        print(len(x_orig[0]))
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
                if df['classes'][i] == classifications['classes_en'][j]:
                    #print(i,df['classes'][i], classifications['classes_en'][j])
                    y.append(classifications['index'][j])

        return x_orig, y_orig, y,classifications,classes

    def NBAnalysis(x_orig, y_orig, y, classifications):
        global num_classes
        num_classes = len(set(y))
        print(num_classes)
        x_train_orig,x_test_orig,y_train_orig,y_test_orig = train_test_split(x_orig,y_orig,test_size=0.2,shuffle=False)
     
        #x =  MLTokenizer.word2vec(x_orig)
        #x =  MLTokenizer.text2vec(x_orig)
        #x =  MLTokenizer.tokenizerTFID(x_orig)  
        #x =  MLTokenizer.tokenizerCount(x_orig)
        #x =  MLTokenizer.tokenizerHashing(x_orig)
        x =  MLTokenizer.tokenizerHybrid(x_orig)


        #print(y)
        #print(len(x))
        #print(len(y))
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,shuffle=False)
        #y_predict_tokenized = mySVM.LinearSVR(x_train,y_train,x_test)
        #y_predict_tokenized = mySVM.LinearSVC(x_train,y_train,x_test)
        #y_predict_tokenized = myClustering.KNN(x_train,y_train,x_test)
        #y_predict_tokenized = myClustering.NC(x_train,y_train,x_test)
        #y_predict_tokenized = myClustering.DBSCANClustering(x_train,y_train,x_test)
        #y_predict_tokenized = myCNN.train(x_train,y_train,x_test,x_test_orig, num_classes,classes)
        #y_predict_tokenized = myMCNN.train(x_train,y_train,x_test,x_test_orig, num_classes,classes)
        #y_predict_tokenized = myAutoEncoder.train(x_train,y_train,x_test,y_test,x_test_orig,num_classes)
        #y_predict_tokenized = myDCNN.train(x_train,y_train,x_test,x_test_orig, num_classes,classes)
        y_predict_tokenized = myDCGAN.train(x_train,y_train,x_test,y_test,x_test_orig,num_classes)
        #print(y_predict_tokenized )
        y_predict=[]
        for i in range(len(y_predict_tokenized )):
            for j in range(len(classifications)):
                #print(y_predict_tokenized[i])
                if y_predict_tokenized[i] == classifications['index'][j]:
                    y_predict.append(classifications['classes_en'][j])
        #print(y_predict)
        return y_predict, y_test,y_predict_tokenized
    def generateOutput(x_orig,y_test,y_predict,y_predict_tokenized):
        y_actual=[]
        for i in range(len(y_test)):
            for j in range(len(classifications)):
                #print(max(y_test[i]))
                if y_test[i] == classifications['index'][j]:
                    y_actual.append(classifications['classes_en'][j])
        begin = -int(np.ceil((0.2*(len(x_orig)))))
        df_test=pd.DataFrame()
        df_test['text'] = x_orig[begin:]
        df_test['class'] = y_actual
        df_test['predicted_class'] = y_predict
        df_test = df_test.reset_index()

        accuracy=0
        for i in range(len(df_test)):
            if df_test['class'][i]==df_test['predicted_class'][i]:
                accuracy=accuracy+1
        print('accuracy is: ',accuracy/len(df_test))
        df_test.to_csv('/userhome/35/xtan/KG_Classification/result_mnnb_20.csv')
        accuracy_score = f1_score(y_test, y_predict_tokenized, average="micro")
        print('f1 score is: ',accuracy_score)
        return df_test, accuracy,accuracy_score

#data_dir = '/userhome/35/xtan/KG_Classification/original_data.csv'
class_dir = '/userhome/35/xtan/KG_Classification/classes.csv'
relations_dir = '/userhome/35/xtan/KG_Classification/df_relations.csv'
num_features = 20
#df = generateRelations.articleGenerateRelations(data_dir)
x_orig, y_orig, y, classifications, classes = textClassifications.dataPreprocessing(relations_dir,class_dir)
y_predict, y_test, y_predict_tokenized = textClassifications.NBAnalysis(x_orig, y_orig, y, classifications)
df_test, accuracy,accuracy_score = textClassifications.generateOutput(x_orig,y_test,y_predict, y_predict_tokenized)
'''
classes_predict=[]

for i in range(len(x_orig)):
  relationships = x_orig[i]
  print(relationships)
  selected_classification = similarityAlgo.similarity(relationships,classes)
  classes_predict.append(selected_classification)

result = pd.DataFrame()
selected_classification = similarityAlgo.similarity_simple(x_orig,classes)
result['relations'] = x_orig
result['classes_actual'] = classes
result['classes_predicted'] = selected_classification
result.to_csv('result_similarityAlgo.csv')
'''
