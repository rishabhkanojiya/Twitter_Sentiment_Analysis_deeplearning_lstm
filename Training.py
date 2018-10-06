
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re


# In[23]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Dropout
from sklearn.model_selection import train_test_split
from keras.models import model_from_json


# In[3]:


import tweepy
from textblob import TextBlob


# In[4]:


# Step 1 - Authenticate
consumer_key= 'OmRf49t1oZTXjHkXRGh6qX0VF'
consumer_secret= 'nEO5x7M4BfKmT7LDjdEwScMyoXyW7AujxxhSLM7kA74kPSQQZc'

access_token='4203546741-JrlHxbM4tlTX9QsaR54TVIq9j6xUXgDzVNdwc6M'
access_token_secret='1486vRM9KoSYDwJ4J16rlDcwBUrmkfw8WaPfkVSJ2uElT'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)


# In[5]:


public_tweets = api.search('@realDonaldTrump')


# In[6]:


data = {'text':[],'sentiment':[]}
data1 = {'text':[],'sentiment':[]}


# In[7]:


for tweet in public_tweets:
    data['text'].append(tweet.text)


# In[8]:


data['text']


# In[9]:


#data[0].split(':')[1]


# In[10]:


data1['text'] = pd.Series(data['text'])


# In[11]:


"""
data2 = {'text':[],'sentiment':[]}
for i in range(len(data1['text'])):
    data2['text'].append(re.sub(r'@[a-zA-Z]+|#|https://','',data1['text'][i]))
data2['text'] = pd.Series(data2['text'])   
data2
"""


# In[12]:


#data1['text']= pd.Series(['I am excited ','I am happy','i am fine','i am sad','i am depressed'])


# In[13]:


data1['text'] = data1['text'].apply(lambda x: x.lower())
data1['text'] = data1['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', ' ', x)))

size = 2200
tokenizer = Tokenizer(num_words=size, split=' ')
tokenizer.fit_on_texts(data1['text'].values)
X2 = tokenizer.texts_to_sequences(data1['text'].values)
#tokenizer.fit_on_texts(pdg.values)
#X2 = tokenizer.texts_to_sequences(pdg.values)
X2 = pad_sequences(X2,maxlen=19)


# In[14]:


X2.shape


# In[15]:


import pandas as pd
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Dropout
from sklearn.model_selection import train_test_split


# In[16]:


data = pd.read_csv("Headline_Trainingdata.csv", sep=',', quotechar='"', header=0, usecols=['text', 'sentiment'])
data['text'] = data['text'].apply(lambda x: x.lower())
data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', ' ', x)))

size = 2200
tokenizer = Tokenizer(num_words=size, split=' ')
tokenizer.fit_on_texts(data['text'].values)
X = tokenizer.texts_to_sequences(data['text'].values)
X = pad_sequences(X)


# In[17]:


model = Sequential()
model.add(Embedding(size, 300, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.5))
model.add(LSTM(128, dropout=0.5, recurrent_dropout=0.5,return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(64, dropout=0.5, recurrent_dropout=0.5,return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(32, dropout=0.5, recurrent_dropout=0.5))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

Y = pd.get_dummies(data['sentiment']).values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
model.fit(X_train, Y_train, epochs=50, batch_size=32, verbose=2)

score, accuracy = model.evaluate(X_test, Y_test, verbose=2)
print("Accuracy: %.2f" % accuracy)


# In[18]:


pred = model.predict_classes(X2)
pred


# In[25]:


model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("model.h5")
print("Saved model to disk")


# In[26]:


json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 

