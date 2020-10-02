import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer,TfidfTransformer,CountVectorizer
import pickle
import numpy as np
import re
from nettoyage import nettoyage
from nltk.stem import SnowballStemmer
from stop_words import get_stop_words
from unidecode import unidecode
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

class Sentiment:
    def __init__(self):
        super().__init__()
        df=pd.read_csv('corpus.csv')
        df['l_review']=df['review'].apply(lambda x:len(x.split(' ')))
        df[(df['rating']<3) & (df['l_review']>5)].describe()
        df=df[df['l_review']>5]
        df['label']=df['rating']
        positif=df[df['label']>3].sample(391)
        negatif=df[df['label']<3]
        Corpus=pd.concat([positif,negatif],ignore_index=True)[['review','label']]

        for ind in Corpus['label'].index:
            if Corpus.loc[ind,'label'] > 3:
                Corpus.loc[ind,'label']=1
            elif Corpus.loc[ind,'label'] < 3:
                Corpus.loc[ind,'label']=0
            pass

        my_stop_word_list = get_stop_words('french')
        s_w=list(set(my_stop_word_list))
        s_w=[elem.lower() for elem in s_w]

        nettoyage(Corpus['review'].loc[1])
        Corpus['review_net']=Corpus['review'].apply(nettoyage)
        vectorizer = TfidfVectorizer()
        vectorizer.fit(Corpus['review_net'])
        X=vectorizer.transform(Corpus['review_net'])
        print(vectorizer.get_feature_names())

        pickle.dump(vectorizer.vocabulary_,open("feature.pkl","wb"))
        self.vectorizer = vectorizer
        y=Corpus['label']
        x_train, x_val, y_train, y_val = train_test_split(X, y, test_size = 0.2)
        cls=LogisticRegression(max_iter=300).fit(x_train,y_train)
        pickle.dump(cls,open("cls.pkl","wb"))
        transformer = TfidfTransformer()
        loaded_vec = CountVectorizer(decode_error="replace",vocabulary=pickle.load(open("feature.pkl", "rb")))
        cls=pickle.load(open("cls.pkl", "rb"))
        self.cls = cls
        self.X = X
        self.y = y
        pass

    def setSentiment(self, sentiment):
        self.sentiment = sentiment
        pass
    
    def entrainement():
        vectorizer = TfidfVectorizer()
        vectorizer.fit(Corpus['review_net'])
        X=vectorizer.transform(Corpus['review_net'])
        pickle.dump(vectorizer.vocabulary_,open("feature.pkl","wb"))
        y=Corpus['label']
        x_train, x_val, y_train, y_val = train_test_split(X, y, test_size = 0.2)
        cls=LogisticRegression(max_iter=300).fit(x_train,y_train)
        pickle.dump(cls,open("cls.pkl","wb"))
        return(cls.score(x_val,y_val))
    
    def evalSentiment(self, phrase):
        self.user = self.vectorizer.transform([nettoyage(phrase)])
        if(self.cls.predict(self.user)[0] == 0.0):
            return 'Désolé pour cette expérience, nous en tiendrons compte'
            pass
        else:
            return 'Merci pour votre retour'
            pass
        pass
    
     




        

