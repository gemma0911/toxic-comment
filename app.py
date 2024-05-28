import streamlit as st
import pickle
import numpy as np
# Load mô hình và vectorizer từ file

with open('train/tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)
with open('train/logistic_regression_model.pkl', 'rb') as f:
    clf = pickle.load(f)

name = st.text_input("Enter your comment")
btn = st.button('Enter')

if btn:
    if name == '':
        st.warning('Vui lòng nhập comment')
    else:
        X_new_tfidf = tfidf_vectorizer.transform([name])
        prediction = clf.predict(X_new_tfidf)
        
        prediction_array = prediction[0]
        print(prediction_array)

        if prediction_array[0] == 1:
            st.warning('Độc hại')
        if prediction_array[1] == 1:
            st.warning('Độc hại nghiêm trọng')
        if prediction_array[2] == 1:
            st.warning('Tục tiểu')
        if prediction_array[3] == 1:
            st.warning('Đe dọa')
        if prediction_array[4] == 1:
            st.warning('Xúc phạm')
        if prediction_array[5] == 1:
            st.warning('Thù ghét')
