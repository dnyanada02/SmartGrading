from flask import Flask,request,render_template,url_for,jsonify
from flask_cors import CORS, cross_origin
import site
import numpy as np
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize,word_tokenize
from gensim.models import Word2Vec
from keras.layers import Embedding, LSTM, Dense, Dropout, Lambda, Flatten, Attention
from keras.models import Sequential, load_model, model_from_config
import keras.backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import cohen_kappa_score
from gensim.models.keyedvectors import KeyedVectors
from keras import backend as K
from tensorflow import keras
from attention import attention
from keras.layers import *
from keras.models import *
from keras import backend as K

# Generating word tokens after removing characters other than alphabets, converting them to lower case and
# removing stopwords from the text'''

def word_tokens(essay_text):
    essay_text = re.sub("[^a-zA-Z]", " ", essay_text)
    words = essay_text.lower().split()
    stop_words = set(stopwords.words("english"))
    words = [w for w in words if not w in stop_words]
    return (words)

# Generating sentence tokens from the essay and finally the word tokens

def sentence_tokens(essay_text):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sent_tokens = tokenizer.tokenize(essay_text.strip())
    sentences = []
    for sent_token in sent_tokens:
        if len(sent_token) > 0:
            sentences.append(word_tokens(sent_token))
    return sentences

# Generating a vector of features

def makeFeatureVec(words, model, num_features):
    vec = np.zeros((num_features,),dtype="float32")
    noOfWords = 0.
    index2word_set = set(model.index_to_key)
    for i in words:
        if i in index2word_set:
            noOfWords += 1
            vec = np.add(vec,model.get_index(i) )        
    vec = np.divide(vec,noOfWords)
    return vec

# Generating word vectors to be used in word2vec model

def getAvgFeatureVecs(essays, model, num_features):
    c=0
    essay_vecs = np.zeros((len(essays),num_features),dtype="float32")
    for i in essays:
        essay_vecs[c] = makeFeatureVec(i, model, num_features)
        c+=1
    return essay_vecs

class attention(Layer):
    def __init__(self,**kwargs):
        super(attention,self).__init__(**kwargs)
 
    def build(self,input_shape):
        self.W=self.add_weight(name='attention_weight', shape=(input_shape[-1],1), 
                               initializer='random_normal', trainable=True)
        self.b=self.add_weight(name='attention_bias', shape=(input_shape[1],1), 
                               initializer='zeros', trainable=True)        
        super(attention, self).build(input_shape)
 
    def call(self,x):
        # Alignment scores. Pass them through tanh function
        e = K.tanh(K.dot(x,self.W)+self.b)
        # Remove dimension of size 1
        e = K.squeeze(e, axis=-1)   
        # Compute the weights
        alpha = K.softmax(e)
        # Reshape to tensorFlow format
        alpha = K.expand_dims(alpha, axis=-1)
        # Compute the context vector
        context = x * alpha
        context = K.sum(context, axis=1)
        return context
    
    
def convertToVec(text):
    content=text
    if len(content) > 20:
        num_features = 300
        model = KeyedVectors.load_word2vec_format("word2vecmodel_Attention_lstm.bin", binary=True)
        clean_test_essays = []
        clean_test_essays.append(word_tokens(content))
        testDataVecs = getAvgFeatureVecs(clean_test_essays, model, num_features )
        testDataVecs = np.array(testDataVecs)
        testDataVecs = np.reshape(testDataVecs, (testDataVecs.shape[0], 1, testDataVecs.shape[1]))
        # with CustomObjectScope({'AttentionLayer': AttentionLayer}):
        #  lstm_model = load_model('/content/Attention_LSTM.h5')
        lstm_model = load_model("Attention_LSTM.h5",custom_objects={'attention': attention})
        preds = lstm_model.predict(testDataVecs)
        print(preds)
        return str(round(preds[0][0]))


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/', methods=['GET','POST'])
@cross_origin()   
def create_task():
    K.clear_session()
    final_text = request.get_json("text")["text"]
    score = convertToVec(final_text)
    K.clear_session()
    return jsonify({'score': score}), 201

if __name__=='__main__':
    app.run(debug=True)
    

