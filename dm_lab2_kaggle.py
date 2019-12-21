# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 01:05:23 2019

@author: Jack

What I try
#Preprocessing:
-remove tagged name
-remove hashtag sharp symbol
-remove unknown tag <LH>, useless puctuation, urls
-wordnet lemmatizing
-nltk English stopwords

#Embedding
-TFIDF
-Word2Vec(training by myself, google pretrain model)
-Word2Vec*TFIDF

#Classifier
-SVM
-Decision Tree
-Deep learning in lab 2


The best result of combination will be:
-remove tagged name
-remove hashtag sharp symbol
-remove unknown tag <LH>, useless puctuation, urls
-wordnet lemmatizing
-TFIDF
-Deep learning in lab 2 with slightly modify


"""
import os
import re
import json
import pickle
from collections import defaultdict
from datetime import datetime

import pandas as pd
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec, KeyedVectors
#from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import recall_score, accuracy_score, f1_score, confusion_matrix
from evaluation import calculate_accuracy
#%% Basic Dense
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense
from keras.callbacks import CSVLogger, EarlyStopping

#data path
base_dir = 'C:/Users/JackChuang/Downloads/kaggle/'
data_dir = base_dir + 'data/'
os.chdir(base_dir)

def generate_submission(pred, index, filename='result.csv'):
    #Input: prediction result, dataframe id, filename
    #Output: log out the result in upload format
    #This function will generate a file which is suitable for kaggle result
    df = pd.DataFrame(pred, index=index, columns=['emotion'])
    df = df.emotion.apply(lambda x: int2emo[x])
    df.to_csv(base_dir+'result/'+filename, header=True)

#%% load data
DEBUG = False
if not os.path.exists(data_dir+'merge.pkl') or DEBUG: # if we didn't load and merge the data before 
    label = pd.read_csv(data_dir+'emotion.csv', index_col='tweet_id')
    sets = pd.read_csv(data_dir+'data_identification.csv', index_col='tweet_id')

    tweets = []
    with open(data_dir+'tweets_DM.json','r') as f:
        for line in f:
            tweets.append(json.loads(line))
    tweets = pd.io.json.json_normalize(tweets, max_level=2) # flatten the json format data

    tweets = tweets.set_index('_source.tweet.tweet_id') # set the index by tweet_id
    tweets.index.name = 'id'
    label.index.name  = 'id'

    raw_data = pd.concat([tweets,label,sets], axis=1, sort=False)
    raw_data.index.name = 'id'
    raw_data.columns = ['_score', '_index', '_crawldate', '_type', 'hashtags', 'text','emotion','sets']
    del tweets, label, sets

    with open(data_dir+'merge.pkl','wb') as f:
        pickle.dump(raw_data ,f)
else: # if we  load and merge the data before 
    with open(data_dir+'merge.pkl','rb') as f: 
        raw_data = pickle.load(f)
#%% data preprocessing
DEBUG = False
if not os.path.exists(data_dir+'merge_split.pkl') or DEBUG: # if we didn't preprocess the data before 
    def remove_redundant(sr):
        lemmatizer = WordNetLemmatizer()
        sr = sr.apply(lambda x: re.sub(r'@\w+',r'@',x)) # remove tagged name        
        sr = sr.apply(lambda x: re.sub(r'<LH>|[,.~\'’"”:;&]|http\S+',' ',x)) # remove unknown tag <LH>, useless puctuation, urls
        sr = sr.apply(lambda x: re.sub(r'#(\w+)',r'\1',x)) # remove hashtag sharp symbol
        #no need TODO: split emoji from words: 2702-27b0 1F600-1F64f 1f300-1f5ff
    #    stopwords = nltk.corpus.stopwords.words('english') #load stopwords in nltk
    #    stopwords.remove('not')
    #    stopwords.remove('nor')
    #    stopwords.remove('no')
        sr = sr.apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.lower().strip().split()]))# if word not in stopwords]))
        return sr
    ## process over all train, valid, test
    raw_data['text'] = remove_redundant(raw_data['text'])
    raw_data['text_token'] = raw_data.text.apply(nltk.word_tokenize)

    ## split train, valid, test sets
    train_x = raw_data[raw_data.sets=='train']
    test_x = raw_data[raw_data.sets=='test']
    del raw_data

    with open(data_dir+'merge_split.pkl','wb') as f:
        pickle.dump([train_x, test_x] ,f)
else:
    with open(data_dir+'merge_split.pkl','rb') as f:
        train_x, test_x = pickle.load(f)

int2emo = train_x.emotion.unique().tolist() # load emotion
emo2int = {v:i for i,v in enumerate(int2emo)}
class_num = len(int2emo) # numbers of emotion
train_x['emotion'] = train_x.emotion.apply(lambda x: emo2int[x])

from sklearn.model_selection import train_test_split
train_x, valid_x, train_y, valid_y = train_test_split(
        train_x, train_x.emotion, test_size=0.1, random_state=1)

train_y_hot, valid_y_hot = map(pd.get_dummies, [train_y, valid_y]) # get one hot result


#%% tfidf
#Get tfidf feature
DEBUG = False
feat_size = 10000
#vectorizer = TfidfVectorizer(min_df=3, max_df=0.9)
if os.path.exists(data_dir+'tfidf{}.pkl'.format(feat_size)) and not DEBUG:
    with open(data_dir+'tfidf{}.pkl'.format(feat_size), 'rb') as f:
        TFIDF = pickle.load(f)
else:
    TFIDF = TfidfVectorizer(max_features=feat_size,tokenizer=nltk.word_tokenize)
    TFIDF.fit(train_x.text)
    with open(data_dir+'tfidf{}.pkl'.format(feat_size), 'wb') as f:
        pickle.dump(TFIDF, f)


train_x_tfidf, valid_x_tfidf, test_x_tfidf = map(TFIDF.transform,[train_x.text,valid_x.text,test_x.text])

#%% w2v-tfidf
#Viewing tfidf as a weighted sum formula, multiply it to w2v model
#
DEBUG = False
feat_size = 100
if os.path.exists(data_dir+'w2v.pkl') and not DEBUG:
    with open(data_dir+'w2v.pkl', 'rb') as f:
        w2v_model = pickle.load(f)
else:
    w2v_model = Word2Vec(sentences=train_x.text_token, size=feat_size)
    with open(data_dir+'w2v.pkl', 'wb') as f:
        pickle.dump(w2v_model, f)

w2v_dict = dict(zip(w2v_model.wv.index2word, w2v_model.wv.vectors))
class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec, tfidf):
        self.word2vec = word2vec
        self.tfidf = tfidf

        max_idf = max(self.tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, self.tfidf.idf_[i]) for w, i in self.tfidf.vocabulary_.items()])

    def transform(self, X):
        return  np.array([
                    np.mean([self.word2vec[w] * self.word2weight[w] if w in self.word2vec else np.zeros(feat_size)
                             for w in words], axis=0)
                for words in X])

w2v_tfidf = TfidfEmbeddingVectorizer(w2v_dict, TFIDF)
#get transformed result
train_x_wtfidf, valid_x_wtfidf, test_x_wtfidf = map(w2v_tfidf.transform,[test_x.text_token,valid_x.text_token,test_x.text_token])

#%% decision tree
clf = DecisionTreeClassifier(class_weight='balanced')
clf.fit(train_x_wtfidf, train_y)

pred = clf.predict(valid_x_wtfidf)
calculate_accuracy(valid_y, pred, int2emo)

pred = clf.predict(test_x_wtfidf)
generate_submission(pred, test_x.index, 'w2v100_tfidf_dtree.csv')

#%% svm
clf = SVC(kernel='linear', class_weight='balanced')
clf.fit(train_x_tfidf, train_y)

pred = clf.predict(valid_x_tfidf)
calculate_accuracy(valid_y, pred, int2emo)

pred = clf.predict(test_x_tfidf)
generate_submission(pred, test_x.index, 'tfidf1000_svm1.csv')

tf.keras.backend.clear_session()
dt = datetime.strftime(datetime.now(),'%m%d%H%M')

#def keras_f1(true,pred): return f1_score(true, pred, average='macro')
#def keras_uar(true,pred): return recall_score(true, pred, average='macro')

# input layer
text_input = Input(train_x_tfidf.shape[1:])
x = Dense(64, activation='relu')(text_input)
x = Dense(64, activation='relu')(x)
outputs = Dense(class_num, activation='softmax')(x)

model = Model(text_input, outputs)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])#,keras_f1,keras_uar])

# training setting
epochs = 20
batch_size = 128

history = model.fit(train_x_tfidf, train_y_hot, 
                    epochs=epochs, 
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data = (valid_x_tfidf, valid_y_hot),
                    callbacks=[CSVLogger(base_dir+'log/training_log_{}.csv'.format(dt))
                               ,EarlyStopping('val_loss',patient=3)
                               ])
print('training finish')

# prediction analysis
pred = np.argmax(model.predict(test_x_tfidf, batch_size=128), axis=1)
generate_submission(pred, test_x.index, 'tfidf10000_dense64.csv')
