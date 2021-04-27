# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
#
#from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
import pickle
#from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv(r'C:\Users\Nancy Meshram\Downloads\final_data.csv')
df=df.drop('Unnamed: 0',axis=1)
df['Over_18'] = df['Over_18']*1
X=df[['Title','Upvote_ratio','Gilded','Over_18','Number_of_Comments','neg','neu','pos','compound']].copy()
y=df.Score.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)
preprocess = ColumnTransformer(
    [('Title_tfidf', TfidfVectorizer(max_features = None,stop_words = 'english', ngram_range=(1,3)), 'Title')],
     remainder='passthrough')
#preprocess.fit(X_train)
#preprocess.transform(X_train)
#preprocess.transform(X_test)

from sklearn.ensemble import RandomForestRegressor
model = make_pipeline(
    preprocess,
   RandomForestRegressor(n_jobs=-1, n_estimators=50, min_samples_leaf=20, random_state = 0))
    
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_pred = model.predict(X_test)

train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
test_r2 = r2_score(y_test, y_pred)
print('Train RMSE: %.4f' % train_rmse)
print('Test RMSE: %.4f' % test_rmse)
print('Test R_2: %.4f' % test_r2)

pickle.dump(model,open("ml.pkl","wb"))
pickle.dump(preprocess,open("ml1.pkl","wb"))


