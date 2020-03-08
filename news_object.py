import os, sys, pickle
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json
import tensorflow as tf
from tensorflow import keras
from keras import backend as K

# Configuration of Keras (Tensorflow sessions)
GPU_OPTIONS = tf.compat.v1.GPUOptions(allow_growth=True)
config = tf.compat.v1.ConfigProto(
    device_count={'GPU': 1},
    intra_op_parallelism_threads=1,
    allow_soft_placement=True, gpu_options=GPU_OPTIONS
    # per_process_gpu_memory_fraction = 0.6
)
# config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.6

session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

sequence_MAX = 1000
words_MAX = 20000
print("loading model .....")

# load json and create model
tf.compat.v1.keras.backend.set_session(session)
m = open('models/model/model.json', 'r')
model_json = m.read()
m.close()
model = model_from_json(model_json)

# load weights into new model
model.load_weights("models/model/model.h5")
print("Loaded model from disk")

# Load pickle files
texts = pickle.load(open('models/text/text.pkl', 'rb')) # Text files
labels_index = pickle.load(open('models/text/labels_index.pkl', 'rb')) # Text labels files
index_to_label_dict = pickle.load(open('models/text/index_to_label_dict.pkl', 'rb')) # labels index dictionary

# Vetorization
tokenizer = Tokenizer(num_words=words_MAX)
tokenizer.fit_on_texts(texts)

# Function to make prediction
def classify_news(news_data):
    with session.graph.as_default():
        tf.compat.v1.keras.backend.set_session(session)
        newsList = []
        newsList.append(news_data)
        test_sequences = tokenizer.texts_to_sequences(newsList)
        test_data = pad_sequences(test_sequences, maxlen=sequence_MAX)
        nn_output = model.predict(test_data)
        i=0
        news_clssification = {}
        for idx in np.argmax(nn_output, axis=1):
            news_clssification[index_to_label_dict[idx]] = newsList[i]
            i = i + 1
        return news_clssification
    K.clear_session()



