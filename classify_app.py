import pandas as pd
import numpy as np
import nltk
import spacy
nlp = spacy.load('en_core_web_lg')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score,classification_report
from sklearn.pipeline import Pipeline
import pickle
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

vec_sub = pickle.load(open('vectorizer_sub.pkl','rb'))
model_sub = pickle.load(open('model_subject.pkl','rb'))
vec_int = pickle.load(open('vectorizer_int.pkl','rb'))
model_int = pickle.load(open('model_intent.pkl','rb'))
vec_senti = pickle.load(open('vectorizer_senti.pkl','rb'))
model_senti = pickle.load(open('model_sentiment.pkl','rb'))
si_obj = SentimentIntensityAnalyzer()

st.write("# Text Classifier App")

x = st.text_input("Enter a text for evaluation")

def get_pos_tags(x):
    doc = nlp(x.lower())
    for token in doc:
        verbs = ','.join([token.text for token in doc if token.pos_ == "VERB"])
        nouns = ','.join([token.text for token in doc if token.pos_ == "NOUN"])
    return verbs,nouns


def task_func(x,vn):
    dict_text = {'query':x.lower()}
    df = pd.DataFrame(dict_text,index = [0])
    df['verbs'] = df['query'].apply(lambda x: vn[0])
    df['nouns'] = df['query'].apply(lambda x: vn[1])
    df['verb_noun'] = df['verbs'].str.cat(df['nouns'],sep = ',')
    df['verb_noun'] = df['verb_noun'].apply(lambda x: ' '.join(x.split(',')))
    df['senti_score'] = df['query'].apply(lambda x: si_obj.polarity_scores(x)["compound"])
    df['Sentiment'] = df["senti_score"].apply(lambda x: "Positive" if x > 0 else ("Negative" if x<0 else "Neutral"))
    subject = model_sub.predict(pd.DataFrame(vec_sub.transform(df['verb_noun'].values).toarray()))
    intent = model_int.predict(pd.DataFrame(vec_int.transform(df['query'].values).toarray()))
    sentiment = df['Sentiment'].values
    text_result = f"Subject : {subject[0]} Intent : {intent[0]} Sentiment : {sentiment[0]}"
    return text_result

if x != '':
    vn = get_pos_tags(x)
    result = task_func(x,vn)
    st.write(result)




    



