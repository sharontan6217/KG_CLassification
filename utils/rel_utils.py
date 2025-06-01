
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
#from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
model_similarity = SentenceTransformer('stsb-roberta-large')



class preprocess():
    def featureExtraction(relations):
        rel=[]
        for relation in relations:
            rel__=[]
            for rel_ in relation:
                rel_value = rel_['head']+' '+rel_['type']+' '+rel_['tail']
                rel__.append(rel_value)
                '''
                for k,v in rel_.items():
                    if k != 'meta':
                        print(k,v)
                        rel__.append(v)

            '''
            rel.append(rel__)
        print(rel)
        return rel
    def featureExtraction_urls(kb):
        relations = kb.relations
        sources = pd.DataFrame.from_dict(kb.sources,orient='columns')
        #print(str(sources.columns[0]))
        df_features = pd.DataFrame()
        for i in range(len(kb.relations)):
            rel_ = kb.relations[i]
            df_ = pd.DataFrame.from_dict(rel_ ,orient='columns')
            df_features = pd.concat([df_features,df_],axis=0)
        df_features['url']=str(sources.columns[0])
        df_features['name']=str(sources.coulmn[1])

        print(df_features)
        return df_features
    def loadClasses(data_dir):
        classifications = pd.read_csv(data_dir,names=['index','classes'])
        print(classifications['classes'][len(classifications)-1])
        classes = classifications['classes']
        classifications['index']=np.arange(0,0+len(classifications))

        return classifications,classes
    def dataClean(text):
        text = str(text)
        text = text.replace('"','')
        text = text.replace("[","")
        text = text.replace("'","")
        text = text.replace("]","")
        return text
    def filterDuplicatedRels(relationships):
        relationships = list(set(relationships))
        for m in range(len(relationships)-1):
            print(m,len(relationships))
            if m<len(relationships):
                encoding_m = model_similarity.encode(relationships[m])
                for n in range(len(relationships)-1):
                    #print('the length of the relationships is: ',len(relationships))
                    if n < len(relationships):
                        #print('m is {},n is {}'.format(m,n))
                        encoding_n = model_similarity.encode(relationships[n])
                        similarity_m = np.dot(encoding_m,encoding_n)/(np.linalg.norm(encoding_m)*np.linalg.norm(encoding_n))
                        if (similarity_m>=0.8 and similarity_m<1.0) == True:
                            #print(relationships[m],relationships[n],similarity_m)
                            del relationships[n]
                            #print('the length of deleted relationships is: ', n,len(relationships))
                    n+=1
                m+=1
        return relationships
    def featuresFiltering(df,num_features):
        df['features']=''
        for i in range(len(df)):
            classification = df['classes'][i]
            relationships = df['relations'][i]
            #relationships = list(set(relationships))
            relationships = preprocess.filterDuplicatedRels(relationships)
            df_similarity=pd.DataFrame()
            similarity = []
            rels = []
            for rel in relationships:
                #print("relationships are ",rel)
                encoding1=model_similarity.encode(rel)
                encoding2=model_similarity.encode(classification)
                similarity_ = np.dot(encoding1,encoding2)/(np.linalg.norm(encoding1)*np.linalg.norm(encoding2))
                print(rel,classification ,similarity_)
                similarity.append(similarity_)
                rels.append(rel)
                #print(len(rels))
                #print(len(similarity))
            df_similarity['similarity']=similarity
            df_similarity['relations']=rels
            df_similarity=df_similarity.sort_values(by=['similarity'],ascending=False)
            print(df_similarity)
            df_similarity = df_similarity.reset_index()
            end = min(num_features,len(df_similarity['relations']))
            relations_top = []
            for j in range(int(end-1)):
              relations_top.append(df_similarity['relations'][j])
            #end = min(num_features,len(df_similarity['relations']))
            #relations_top = preprocess.filterDuplicatedRels(relations_top)[:end]
            df['features'][i] = relations_top
            df['relations'][i] = relationships
            '''
            for j in range(num_features):
                col_name = 'feature_'+str(j)
                df[col_name]=''
                df[col_name][i] = df_similarity['relations'][j]
            '''
        print(df)
        return df
