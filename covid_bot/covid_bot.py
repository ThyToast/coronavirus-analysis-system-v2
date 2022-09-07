import pickle
import re
import string
import joblib
import numpy as np
import pandas as pd
import pyttsx3
import streamlit as st

from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


def get_text():
    input_text = st.text_input("Type any COVID-19 questions here: ")
    df_input = pd.DataFrame([input_text], columns=['questions'])
    return df_input


class CovidBot:
    model = load_model('covid_bot/datasets/model-v1.h5')
    tokenizer_t = joblib.load('covid_bot/datasets/tokenizer_t.pkl')
    vocab = joblib.load('covid_bot/datasets/vocab.pkl')
    df2 = pd.read_csv('covid_bot/datasets/response.csv')

    def get_pred(model, encoded_input):
        pred = np.argmax(model.predict(encoded_input))
        return pred

    def bot_precaution(df_input, pred):
        words = df_input.questions[0].split()
        if len([w for w in words if w in CovidBot.vocab]) == 0:
            pred = 1
        return pred

    def get_response(df2, pred):
        with open('covid_bot/datasets/mapper.p', 'rb') as fp:
            mapper = pickle.load(fp)

        upper_bound = df2.groupby('labels').get_group(mapper[pred]).shape[0]
        r = np.random.randint(0, upper_bound)
        responses = list(df2.groupby('labels').get_group(mapper[pred]).response)
        return responses[r]

    def bot_response(response, ):
        return response

    def remove_stop_words_for_input(tokenizer, df, feature):
        doc_without_stopwords = []
        entry = df[feature][0]
        tokens = tokenizer(entry)
        doc_without_stopwords.append(' '.join(tokens))
        df[feature] = doc_without_stopwords
        return df

    def encode_input_text(tokenizer_t, df, feature):
        t = tokenizer_t
        entry = entry = [df[feature][0]]
        encoded = t.texts_to_sequences(entry)
        padded = pad_sequences(encoded, maxlen=7, padding='post')
        return padded

    def tokenizer(entry):
        tokens = entry.split()
        lemmatizer = WordNetLemmatizer()
        re_punc = re.compile('[%s]' % re.escape(string.punctuation))
        tokens = [re_punc.sub('', w) for w in tokens]
        tokens = [word for word in tokens if word.isalpha()]
        tokens = [lemmatizer.lemmatize(w.lower()) for w in tokens]
        #     stop_words = set(stopwords.words('english'))
        #     tokens = [w for w in tokens if not w in stop_words]
        tokens = [word.lower() for word in tokens if len(word) > 1]
        return tokens

    def botResponse(user_input):
        tokenizer_t = joblib.load('covid_bot/datasets/tokenizer_t.pkl')
        df_input = CovidBot.remove_stop_words_for_input(CovidBot.tokenizer, user_input, 'questions')
        encoded_input = CovidBot.encode_input_text(tokenizer_t, df_input, 'questions')

        pred = CovidBot.get_pred(CovidBot.model, encoded_input)
        pred = CovidBot.bot_precaution(df_input, pred)

        response = CovidBot.get_response(CovidBot.df2, pred)
        response = CovidBot.bot_response(response)

        if st.session_state["is_startup"]:
            response = "Hi, I'm happy to have you here \nI hope you're doing well today :)"
            st.session_state["is_startup"] = False
            return response

        else:
            return response

    def speak(user_input):
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[1].id)
        engine.say(user_input)
        engine.runAndWait()
