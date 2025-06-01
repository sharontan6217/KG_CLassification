import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer, util

model_similarity = SentenceTransformer('stsb-roberta-large')
class similarityAlgo():
    def similarity(relationships,classes):
        #print(len(classes))
        #relationships = list(set(relationships))
        #relationship = preprocess.filterDuplicatedRels(relationships)
        print(len(relationships))
        df_similarity = pd.DataFrame()
        df_similarity['class']=classes
        df_similarity['index']=0
        scores=[]
        for m in range(len(df_similarity['class'])):
            similarity = []
            df_similarity['index'][m] = m
            for n in range(len(relationships)):
                sentence1=relationships[n]
                sentence2=classes[m]
                encoding1=model_similarity.encode(sentence1)
                encoding2=model_similarity.encode(sentence2)
                similarity_ = np.dot(encoding1,encoding2)/(np.linalg.norm(encoding1)*np.linalg.norm(encoding2))
                #print(sentence1,sentence2,similarity)
                if similarity_ >=0.4:
                    print(sentence1,sentence2,similarity_)
                    similarity.append(similarity_ )
            similarity.sort(reverse=True)
            similarity_top = similarity[:min(5,len(similarity))]
            score = sum(similarity_top)
            #print('top {} similarity scores are: {}, total score is {}'.format(min(3,len(similarity)),similarity_top,score))
            scores.append(score)
        df_similarity['scores']=scores
        df_similarity=df_similarity.sort_values(by='scores',ascending=False)
        df_similarity=df_similarity.reset_index()
        #print(df_similarity)
        selected_classification = df_similarity['index'][0]
        #print(df_similarity[:5])
        print('---------------------------selected classification-----------------')
        print(selected_classification,df_similarity['class'][0])

        return selected_classification

    def similarity_simple(relationships,classes):
        #print(len(classes))
        #relationships = list(set(relationships))
        #relationship = preprocess.filterDuplicatedRels(relationships)
        print(len(relationships))
        selected_classification=[]
        for n in range(len(relationships)):
            similarity = []
            df_similarity = pd.DataFrame()
            df_similarity['class']=classes
            scores=[]
            for m in range(len(df_similarity['class'])):
                sentence1=relationships[n]
                #sentence1=preprocess.dataClean(str(sentence1))
                sentence2=classes[m]
                encoding1=model_similarity.encode(sentence1)
                encoding2=model_similarity.encode(sentence2)
                similarity_ = np.dot(encoding1,encoding2)/(np.linalg.norm(encoding1)*np.linalg.norm(encoding2))
                print(sentence1,sentence2,similarity)
                #if similarity_ >=0.4:
                print(sentence1,sentence2,similarity_)
                similarity.append(similarity_ )
            df_similarity['scores']=similarity
            df_similarity=df_similarity.sort_values(by='scores',ascending=False)
            df_similarity=df_similarity.reset_index()
            #print(df_similarity)
            selected_classification_ = df_similarity['class'][0]
            #print(df_similarity[:5])
            print('---------------------------selected classification-----------------')
            print(selected_classification_)
            selected_classification.append(selected_classification_)

        return selected_classification
