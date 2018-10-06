
# coding: utf-8

# In[127]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Dropout
from sklearn.model_selection import train_test_split
from keras.models import model_from_json


# In[95]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from tweepy import OAuthHandler
from tweepy import API
from tweepy import Cursor
from datetime import datetime, date, time, timedelta
from collections import Counter
import sys


# In[96]:


# Step 1 - Authenticate
consumer_key= 'consumer_key'
consumer_secret= 'consumer_secret'

access_token='access_token'
access_token_secret='access_token_secret'
 
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
auth_api = API(auth)


# In[117]:


account_list = ['@realDonaldTrump'] 


# In[170]:


op = {}


# In[171]:


if len(account_list) > 0:
  for target in account_list:
    item = auth_api.get_user(target)
    op["name "]  = item.name
    op["screen_name "] = item.screen_name
    op["description "] = item.description
    op["statuses_count "] = str(item.statuses_count)
    op["friends_count " ] =str(item.friends_count)
    op["followers_count "] = str(item.followers_count)
tweets = item.statuses_count
account_created_date = item.created_at
delta = datetime.utcnow() - account_created_date
account_age_days = delta.days
op["Account age (in days) "] = str(account_age_days)
if account_age_days > 0:
    op["Average tweets per day "] = ("%.2f"%(float(tweets)/float(account_age_days)))
hashtags = []
mentions = []
tweet_count = 0
end_date = datetime.utcnow() - timedelta(days=30)
for status in Cursor(auth_api.user_timeline, id=target).items():
  tweet_count += 1
  if hasattr(status, "entities"):
    entities = status.entities
    if "hashtags" in entities:
      for ent in entities["hashtags"]:
        if ent is not None:
          if "text" in ent:
            hashtag = ent["text"]
            if hashtag is not None:
              hashtags.append(hashtag)
    if "user_mentions" in entities:
      for ent in entities["user_mentions"]:
        if ent is not None:
          if "screen_name" in ent:
            name = ent["screen_name"]
            if name is not None:
              mentions.append(name)
    if status.created_at < end_date:
        break


op["All done. Processed "]  =  (str(tweet_count) + " tweets.")


# In[172]:


public_tweets = auth_api.search(account_list)


# In[173]:


data = {'text':[],'sentiment':[]}
data1 = {'text':[],'sentiment':[]}


# In[174]:


for tweet in public_tweets:
    data['text'].append(tweet.text)


# In[175]:


data1['text'] = pd.Series(data['text'])


# In[176]:


data1['text'] = data1['text'].apply(lambda x: x.lower())
data1['text'] = data1['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', ' ', x)))

size = 2200
tokenizer = Tokenizer(num_words=size, split=' ')
tokenizer.fit_on_texts(data1['text'].values)
X2 = tokenizer.texts_to_sequences(data1['text'].values)
X2 = pad_sequences(X2,maxlen=19)


# In[177]:


json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("model.h5")


# In[178]:


pred = loaded_model.predict_classes(X2)
op['Pred'] = pred.tolist()


# In[179]:


op


# In[185]:


import json

r= op 
r = json.dumps(r)
loaded_r = json.loads(r)
loaded_r
with open('loaded_r.json', 'w') as outfile:
    json.dump(loaded_r, outfile)

