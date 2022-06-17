import flask
import os
import joblib
import json
import pickle
from flask import request
from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, Permute, Dropout
from keras.layers import add, dot, concatenate
from keras.layers import LSTM

import numpy as np
import pandas as pd
with open('train_qa.txt', 'rb') as f:
    train_data = pickle.load(f)
with open('test_qa.txt', 'rb') as f:
    test_data = pickle.load(f)
all_data = test_data + train_data
all_story_lens = [len(data[0]) for data in all_data]
max_story_len = max(all_story_lens)
max_question_len = max([len(data[1]) for data in all_data])
vocab=set()
for story,question,answer in all_data:
    vocab = vocab.union(set(story))
    vocab = vocab.union(set(question))
vocab.add('no')
vocab.add('yes')
vocab_len = len(vocab)+1
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(filters=[])
tokenizer.fit_on_texts(vocab)

def vectorize_stories(data, word_index=tokenizer.word_index, max_story_len=max_story_len, max_question_len=max_question_len):

    tokenizer = Tokenizer(filters=[])
    tokenizer.fit_on_texts(vocab)
    
    X= []
    Xq= []
    for story, query in data:
        x= [word_index[word.lower()] for word in story] 
        xq= [word_index[word.lower()] for word in query]
        X.append(x)
        Xq.append(xq)
    return (pad_sequences(X, maxlen=max_story_len),pad_sequences(Xq,maxlen=max_question_len))

app= flask.Flask(__name__)
app.config["Debug"]= True
from flask_cors import CORS
CORS(app)
@app.route('/', methods = ['GET'])
def default():
    return " This is an API to deploy a chatbot."

@app.route('/Chatbot', methods = ['GET'])
def chatbot():
    return "A Chatbot is a computer program that stimulates and processes human conversation (either written or spoken), allowing humans to interact with digital devices as if they are communicating with real person"

# Q/A from UI part.
@app.route('/Prediction', methods = ['GET','POST'])
def prediction():
    #model = keras.models.load_model('Chatbot_deployment_model.pkl')
    model = keras.models.load_model('Chatbot_deployment_model.h5')
    #Chatbot_deployment_model.h5
    my_story = request.args.get('my_story')
    my_question = request.args.get('my_question')
    mydata = [(my_story.split(),my_question.split())]
    my_story_vec,my_question_vec = vectorize_stories(mydata)
    pred_results = model.predict(([ my_story_vec, my_question_vec]))
    val_max = np.argmax(pred_results[0])
    for key, val in tokenizer.word_index.items():
        if val==val_max:
            return 'Yes' #+ ' | my_story:'+ str(my_story)+ ' | '+ ', my_question:'+ str(my_question)
        else:
            return 'No'#+ ' | my_story:'+ str(my_story)+ ' | '+ ', my_question:'+ str(my_question)

if __name__=='__main__':
    app.run(host = '0.0.0.0', port =8080) # This is mandatory for cloud.