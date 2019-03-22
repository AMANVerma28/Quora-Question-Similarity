import pickle
import re
import itertools
import pandas as pd
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import keras.backend as K

def exponent_neg_manhattan_distance(left, right):
    ''' Helper function for the similarity estimate of the LSTMs outputs'''
    return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))

def text_to_word_list(text):
    ''' 
    Pre process and convert texts to a list of words 
    input: str
    output: list of cleaned word
    '''
    
    text = str(text)
    text = text.lower()

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub("quikly","quickly", text)

    text = text.split()

    return text

def load_tokenizer():
	TOKENIZER_FILE = "/home/aman/Downloads/tokenizer"
	tokenizer_file = open(TOKENIZER_FILE, "rb")
	tokenizer = pickle.load(tokenizer_file)
	tokenizer_file.close()
	return tokenizer

def load_model_test():
	MODEL_SAVING_DIR = "/home/aman/Downloads/saved_model_v2.h5"
	model = load_model(MODEL_SAVING_DIR, custom_objects={'exponent_neg_manhattan_distance': exponent_neg_manhattan_distance})
	return model

def test_on_example(q1, q2, tokenizer, model):
    cleaned_q1 = []
    cleaned_q2 = []
    q1 = [q1]
    q2 = [q2]
    # if 'model' not in globals():
    # 	print("Loading")
    # 	tokenizer, model = load_resources()
    for q in list(q1):
        q1_words = text_to_word_list(q)
        cleaned_q1.append(" ".join(q1_words))
    
    for q in list(q2):
        q2_words = text_to_word_list(q)
        cleaned_q2.append(" ".join(q2_words))
        
    tokenized_q1 = tokenizer.texts_to_sequences(cleaned_q1)
    tokenized_q2 = tokenizer.texts_to_sequences(cleaned_q2)
    
    to_test = pd.DataFrame({'question1':list(tokenized_q1), 'question2':list(tokenized_q2)})
    
    X_to_test = {'left':to_test.question1,'right':to_test.question2}
    # zero padding
    for dataset, side in itertools.product([X_to_test],['left','right']):
        dataset[side] = pad_sequences(dataset[side], maxlen = 25)

    score = model.predict([X_to_test["left"], X_to_test["right"]])
    return round(score[0][0]*100, 2)
    # for s in score:
    #     if s >= 0.5:
    #         print("The sentences are same\nConfidence = ","{:.4f}".format(s[0]))
    #     else:
    #         print("The sentences are different\nConfidence = ","{:.4f}".format(s[0]))


