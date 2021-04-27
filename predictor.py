# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 23:06:47 2021

@author: Nancy Meshram
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
model=pickle.load(open("ml.pkl","rb"))
cv=pickle.load(open("ml1.pkl","rb"))

def predict(Title,Upvote_ratio,Glided,Over_18,Number_of_Comments,neg,neu,pos,compound):
    text=cv.transform([[Title,Upvote_ratio,Glided,Over_18,Number_of_Comments,neg,neu,pos,compound]]).toarray()
    vect=pd.DataFrame(text)
    data=np.array([[Upvote_ratio,Glided,Over_18,Number_of_Comments,neg,neu,pos,compound]]).astype(float)
    text = st.text_input('URL')
    data1=pd.DataFrame(data)
    newdata=pd.concat([data1,vect],axis=1)
    prediction=model.predict(newdata)
    pred='{0:.{1}f}'.format(prediction[0][0],2)
    return float(pred)


def main():
    st.title("Popularity Post Prediction")
    html_temp = """
    <div style="background-color:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Post Prediction ML App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    Title = st.text_input('Title',"Type Here")
    Upvote_ratio = st.text_input("Upvote_ratio","Type Here")
    Glided = st.text_input("Glided","Type Here")
    Over_18 = st.text_input("Over_18","Type Here")
    Number_of_Comments = st.text_input("Number_of_Comments","Type Here")
    neg = st.text_input("neg","Type Here")
    neu = st.text_input("neu","Type Here")
    pos = st.text_input("pos","Type Here")
    compound = st.text_input("compound","Type Here")
    
    if st.button("Predict"):
        output=predict(Title,Upvote_ratio,Glided,Over_18,Number_of_Comments,neg,neu,pos,compound)
        st.success('The probability of the post is {}'.format(output))
    

        
if __name__=='__main__':
    main()