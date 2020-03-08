import os, sys, pickle
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json
import tensorflow as tf
from tensorflow import keras
from keras import backend as K


GPU_OPTIONS = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(
    device_count={'GPU': 1},
    intra_op_parallelism_threads=1,
    allow_soft_placement=True, gpu_options=GPU_OPTIONS)
    
session = tf.Session(config=config)
keras.backend.set_session(session)

def init(md, wt):
    # Configuration of Keras (Tensorflow sessions)
    

    print("loading model .....")

    # load json and create model
    keras.backend.set_session(session)
    m = open(md, 'r')
    model_json = m.read()
    m.close()
    model = model_from_json(model_json)

    # load weights into new model
    model.load_weights(wt)
    print("Loaded model from disk")

    return model

def predict(news_data, texts, model, index_dict):
    sequence_MAX = 1000
    words_MAX = 20000
    tokenizer = Tokenizer(num_words=words_MAX)
    tokenizer.fit_on_texts(texts)
    with session.graph.as_default():
        keras.backend.set_session(session)
        newsList = []
        newsList.append(news_data)
        test_sequences = tokenizer.texts_to_sequences(newsList)
        test_data = pad_sequences(test_sequences, maxlen=sequence_MAX)
        nn_output = model.predict(test_data)
        i=0
        news_clssification = {}
        for idx in np.argmax(nn_output, axis=1):
            news_clssification[index_dict[idx]] = newsList[i]
            i = i + 1
        return news_clssification
    K.clear_session()


class GetText():
    def load_text(self, dir):
        # 
        # words_MAX = 20000
        # Text files
        texts = pickle.load(open(dir, 'rb'))

        # tokenizer = Tokenizer(num_words=words_MAX)
        
        return texts

    def load_label_index(self, dir):
        # Text labels files
        labels_index = pickle.load(open(dir, 'rb'))
        return labels_index

    def load_index_label_dict(self, dir):
        # labels index dictionary
        index_to_label_dict = pickle.load(open(dir, 'rb')) 
        return index_to_label_dict