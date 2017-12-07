import csv
import tensorflow as tf
import keras
from pandas import DataFrame
import pandas
import numpy as np
from keras.datasets import imdb
from keras import *
from keras.preprocessing import sequence
from keras.layers import *
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer

df = pandas.read_csv("joci.csv")
#Prop = df['Property']
#subt = df[['Pred.Token', 'Pred.Lemma', 'Arg.Phrase', 'Property']]
subt2 = df.iloc[0, :]
print (subt2)
print (list(df))
print (np.array([[1], [2]]).T[0])
np.random.seed(7)
top_words = 1000
#(xtr, ytr), (xt, yt) = imdb.load_data(num_words=top_words)
#print (xtr[0])
df = df.values
xtr1 = df[0:35000,0:1].T[0].tolist()
xtr2 = df[0:35000,1:2].T[0].tolist()
ytr = df[0:35000,2:3].T[0].tolist()
xt1 = df[35000:, 0:1].T[0].tolist()
xt2 = df[35000:, 1:2].T[0].tolist()
yt = df[35000:, 2:3].T[0].tolist()
data = xtr1 + xtr2 + xt1 + xt2
print (len(data))
tknzr = Tokenizer(num_words=None,
                  filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                  lower=True,
                  split=" ",
                  char_level=False)
tknzr.fit_on_texts(data)
xtr1 = tknzr.texts_to_sequences(xtr1)
xtr2 = tknzr.texts_to_sequences(xtr2)
xt1 = tknzr.texts_to_sequences(xt1)
xt2 = tknzr.texts_to_sequences(xt2)
print (xtr1)
#xtr = sequence.pad_sequences(xtr, 500)
#xt = sequence.pad_sequences(xt, 500)

emb_vl = 16

inp1 = Input(shape=(None, num_encoder_tokens), name='premise')
inp1 = Embedding(top_words, emb_vl, input_length=500)(inp1)
inp2 = Input(shape=(None, num_encoder_tokens), name='conclusion')
inp2 = Embedding(top_words, emb_vl, input_length=500)(inp2)
merge = keras.layers.concatenate([inp1, inp2])
merge = Dense(256, activation='relu')(merge)
merge = LSTM(128)(merge)
#merge = LSTM(128)(merge)
output = Dense(1, activation='sigmoid')(merge)

#model = Sequential()
#model.add(Embedding(top_words, emb_vl, input_length=500))
#model.add(LSTM(50))
#model.add(Dense(1, activation="sigmoid"))
model = Model(inputs=[inp1, inp2], outputs=[output])
model.compile(loss="binary_crossentropy", optimizer="ADAM", metrics=['accuracy'])
print(model.summary())
#model.fit(xtr, ytr, validation_data=(xt, yt), epochs=3, batch_size=64)

#scores = model.evaluate(xt, yt, verbose=0)
#print (scores[1]*100)
