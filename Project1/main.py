################################ Imports ######################################
import math

import spacy
from datasets import load_dataset
from datasets import get_dataset_split_names
import copy
import numpy as np, pandas as pd

################################ Constants ####################################
SENTENCE_TASK2 = ("I have a house in").split()  # str -> list
FIRST_SENTENCE = ("START Brad Pitt was born in Oklahoma").split()  # str -> list
SECOND_SENTENCE = ("START The actor was born in USA").split()  # str -> list
test = ("START The game began development in").split()


################################ Main functions ###############################

def load_data():
    df = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    nlp = spacy.load("en_core_web_sm")
    # text = df['text'][:10000]
    text = df['text']
    return text, nlp


def procces_data(text, nlp):
    words = set()  # contain all distinct lemmas in training corpus
    docs = []
    for sentence in text:
        # sentence = "START " + sentence  # I put spaces here be aware
        doc = nlp(sentence)
        lemma_docs = [token.lemma_ for token in doc if token.is_alpha]
        if lemma_docs:
            lemma_lower_docs = convert_to_lower_case(lemma_docs)  # convert to array and then to lowercase
            docs.append(list(lemma_lower_docs))
            for token in lemma_lower_docs:
                words.add(token)
    return docs, words


################################ TASK 1 ###############################

def train_unigram_model(docs):
    unigram = dict()
    for doc in docs:
        for token in doc:
            if token in unigram.keys():
                unigram[token] += 1
            else:
                unigram[token] = 1
    unigram_model = pd.DataFrame.from_dict(unigram, orient='index')
    unigram_model = unigram_model / np.sum(unigram_model.values)
    return unigram_model


def train_bigram_model(docs, words):
    words_arr = np.array(["START"] + list(words))
    b_gram_model = pd.DataFrame(data=np.zeros((words_arr.shape[0], words_arr.shape[0])), index=words_arr,
                                columns=words_arr)
    b_gram = dict()  # creating a nested dictionary. to hold biagram
    # create b_gram dict and fills it
    for doc in docs:
        cur_doc = ["START"] + doc
        for i in range(len(cur_doc) - 1):
            word = cur_doc[i]
            next_word = cur_doc[i + 1]
            if word not in b_gram.keys():
                b_gram[word] = dict()
            if next_word in b_gram[word].keys():
                b_gram[word][next_word] += 1
            else:
                b_gram[word][next_word] = 1

    # fills b_gram_model with propabilites
    for word in b_gram.keys():
        cur_dict = b_gram[word]
        for next_word in cur_dict.keys():
            value = cur_dict[next_word]
            # b_gram_model[next_word][word] = value #TODO  why its working?
            b_gram_model.loc[word][next_word] = value  # TODO  why its working?
        b_gram_model.loc[word] /= np.sum(list(cur_dict.values()))

    return b_gram_model


################################ TASK 2 ###############################
def continue_sentence(b_gram_model):
    prev_word = SENTENCE_TASK2[-1]
    index = np.argmax(b_gram_model.loc[prev_word])
    predicted_word = b_gram_model.columns[index]
    return predicted_word


################################ TASK 3 ###############################
def predict_sentence_probabilty_b_gram(sentence, b_gram_model, nlp):
    prob = 1
    sentence = nlp(str(convert_to_lower_case(sentence)))

    sentence = [token.lemma_ for token in sentence if token.is_alpha]
    for i in range(len(sentence) - 1):
        word = sentence[i]
        if i == 0:
            word = "START"
        next_word = sentence[i + 1]
        if word not in b_gram_model.keys() or next_word not in b_gram_model.loc[word].keys():
            return -math.inf
        temp = b_gram_model.loc[word][next_word]  # TODO WHY LIKE THIS
        prob *= temp
    return prob


################################ TASK 4 ###############################

# def linear_interpolation(sentence, unigram_model, b_gram_model, nlp):
#     bigram_lambda = 2 / 3
#     unigram_lambda = 1 / 3
#     unigram_prob =  predict_sentence_probabilty_unigram(sentence, unigram_model, nlp)
#     # unigram_prob = log_probability(unigram_prob)
#     b_gram_prob = predict_sentence_probabilty_b_gram(sentence, b_gram_model, nlp)
#     # b_gram_prob = log_probability(b_gram_prob)
#     prediction = log_probability(unigram_lambda * unigram_prob+ \
#                  bigram_lambda * b_gram_prob)
#     return prediction

def linear_interpolation(sentence, unigram_model, b_gram_model, nlp):
    bigram_lambda = 2 / 3
    unigram_lambda = 1 / 3
    prob = 0
    sentence = nlp(str(convert_to_lower_case(sentence)))
    sentence = [token.lemma_ for token in sentence if token.is_alpha]
    for i in range(len(sentence) - 1):
        # b_gram
        word = sentence[i]
        if i == 0:
            word = "START"
        next_word = sentence[i + 1]
        if word not in b_gram_model.keys() or next_word not in b_gram_model.loc[word].keys():
            b_gram_prob = 0
        else:
            b_gram_prob = bigram_lambda *  b_gram_model.loc[word][next_word]

        # unigram
        if next_word not in unigram_model.index:
            unigram_prob = 0
        else:
            unigram_prob = unigram_lambda * unigram_model.loc[next_word][0].astype(float)

        prob += log_probability(unigram_prob + b_gram_prob)
    return prob #comment

def predict_perplexity3(b_gram_model, nlp):
    first_prob = predict_sentence_probabilty_b_gram(FIRST_SENTENCE, b_gram_model, nlp)
    second_prob = predict_sentence_probabilty_b_gram(
        SECOND_SENTENCE, b_gram_model, nlp)
    first_prob = log_probability(first_prob)
    second_prob = log_probability(second_prob)
    held_out_data = (first_prob + second_prob) / 2
    perplexity = 2 ** -held_out_data
    return perplexity

def predict_perplexity4(b_gram_model,unigram_model, nlp):
    held_out_data = (linear_interpolation(FIRST_SENTENCE, unigram_model, b_gram_model, nlp) + linear_interpolation(
    SECOND_SENTENCE, unigram_model, b_gram_model, nlp)) / (len(FIRST_SENTENCE) + len(SECOND_SENTENCE) -2) #-2 because we added START to each sentence
    perplexity = math.exp(-held_out_data)
    return perplexity

################################ Helpers ######################################
def convert_to_lower_case(sentence):
    return np.char.lower(np.array(sentence))


def predict_sentence_probabilty_unigram(sentence, unigram_model, nlp):
    prob = 1
    sentence = nlp(str(convert_to_lower_case(sentence)))

    sentence = [token.lemma_ for token in sentence if token.is_alpha]
    for word in sentence:
        if word not in unigram_model.index:
            return -math.inf
        temp = unigram_model.loc[word][0]
        prob *= temp
    return prob


def log_probability(prob):
    if prob == 0:
        return -math.inf
    else:
        return math.log(prob)

if __name__ == '__main__':
    # load data

    text, nlp = load_data()
    docs, words = procces_data(text, nlp)

    # TASK 1
    unigram_model = train_unigram_model(docs)
    b_gram_model = train_bigram_model(docs, words)

    # TASK 2
    predicted_word = continue_sentence(b_gram_model)
    print("### TASK 2 ###")
    print("Predicted word: " + predicted_word + "\n")

    # TASK 3
    print("### TASK 3 ###")
    # 3A
    prob3a1 = predict_sentence_probabilty_b_gram(FIRST_SENTENCE, b_gram_model, nlp)
    prob3a1 = log_probability(prob3a1)
    prob3a2 = predict_sentence_probabilty_b_gram(SECOND_SENTENCE, b_gram_model, nlp)
    prob3a2 = log_probability(prob3a2)
    print("Probability of first sentence: " + str(prob3a1))
    print("Probability of second sentence: " + str(prob3a2))
    # 3B
    perplexity3B = predict_perplexity3(b_gram_model, nlp)
    print("Perplexity of both sentences: " + str(perplexity3B) + "\n")

    print(perplexity3B)
    # TASK 4
    print("### TASK 4 ###")
    prob4A = linear_interpolation(FIRST_SENTENCE, unigram_model, b_gram_model, nlp)
    prob4B = linear_interpolation(SECOND_SENTENCE, unigram_model, b_gram_model, nlp)
    perplexity4 = predict_perplexity4(b_gram_model, unigram_model, nlp)
    print("Linear interpolation smoothing for first sentence: " + str(prob4A))
    print("Linear interpolation smoothing for second sentence: " + str(prob4B))
    print("Perplexity of both sentences: " + str(perplexity4))


