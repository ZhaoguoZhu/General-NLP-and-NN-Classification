# packages necessary for text preprocess
import matplotlib.pyplot as plt
import plotly.express as px
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import re
import string
import nltk
import numpy as np 
import pandas as pd
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from gensim.models import word2vec

# read in data, the training data will be labeled as Train_Data
Train_Data = pd.read_csv('./data/train.csv')
Road_Test_Data = pd.read_csv('./data/test.csv')

# stopwords focus on english, stop_words
stop_words = stopwords.words('english')

# function clean_text to make all string input uniform and ready for digitalization:
def clean(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text) # remove all links
    text = re.sub("@[A-Za-z0-9_]+","", text) # remove all mention
    text = re.sub("#[A-Za-z0-9_]+","", text) # remove all hashtag
    text = re.sub(r'[0-9]', '', text) # remove all digit
    text = text.translate(str.maketrans("","", string.punctuation)) # remove punctuation
    text = text.split() # split the text to remove stop words
    text = [word for word in text if not word in stop_words] # remove stop words
    text = ' '.join(text)# rejoin as a whole string
    text = re.sub("[^a-zA-Z0-9]+", " ",text) # remove non english characters
    text = text.split() # clean all additional white space
    text = ' '.join(text)# rejoin as a whole string
    return text

# clean the text
Train_Data['text'] = Train_Data['text'].apply(clean)
Road_Test_Data['text'] = Road_Test_Data['text'].apply(clean)

# Inspecting the text and gather their length information
Train_Data['length']=[len(x.split()) for x in Train_Data['text'].tolist()]

# split train and test
feature, label= Train_Data['text'], Train_Data['target']
X_train, X_test, Y_train, Y_test = train_test_split(feature, label, train_size=0.9,random_state=42,shuffle = True,stratify=label)

# save as numpy data 
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
Y_train = Y_train.to_numpy()
Y_test = Y_test.to_numpy()


# Tokenization
MAX_VOCAB_LENGTH = 20000
AVERAGE_WORD_LENGTH = 8 # average length observed in Train_Data['length']
text_vectorizer = TextVectorization(max_tokens=MAX_VOCAB_LENGTH, 
                                       split="whitespace",
                                       ngrams=None,
                                       output_mode='int',
                                       output_sequence_length=AVERAGE_WORD_LENGTH,
                                       pad_to_max_tokens=True)
text_vectorizer.adapt(X_train)

# Embedding
embedding = Embedding(input_dim=MAX_VOCAB_LENGTH, output_dim=256, input_length=AVERAGE_WORD_LENGTH)

# Training with simple neural network
model_1 = tf.keras.Sequential()
model_1.add(text_vectorizer)
model_1.add(embedding)
model_1.add(tf.keras.layers.Flatten())
model_1.add(tf.keras.layers.Dense(128,activation='relu',activity_regularizer=tf.keras.regularizers.L2(1)))
model_1.add(tf.keras.layers.Dropout(0.2))
model_1.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model_1.compile(loss="binary_crossentropy",optimizer="adam", metrics=["accuracy"])
model_1.summary()

model_1_history = model_1.fit(x=X_train,
                              y=Y_train,
                              epochs=15,
                              validation_split=0.2,verbose=2)

model_1.evaluate(X_test,Y_test)

Road_Test = Road_Test_Data['text']
#model_1.predict(Road_Test)

