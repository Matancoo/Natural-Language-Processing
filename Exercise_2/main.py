################################ Imports ######################################
import numpy as np, pandas as pd
import nltk
import re
from nltk.corpus import brown
from sklearn.metrics import confusion_matrix
import plotly.express as px
nltk.download("brown")
from sklearn.model_selection import train_test_split


################################ Constants ####################################

################################ TASK a ###############################
def load_data():
    data = brown.tagged_sents(categories="news")
    train_set, test_set = train_test_split(data, test_size=0.1)

    return train_set, test_set, data


################################ TASK b ###############################
######### TASK b-i #########

def init_model(possible_tag_for_words, words_set):
    '''

    :param possible_tag_for_words:
    :param words_set:
    :return: model: pd of all words and most likely tag
    '''
    model = pd.DataFrame(data=np.zeros(len(words_set)), index=list(words_set))
    model[0] = "NN"
    for word in model.index:
        if word in possible_tag_for_words.keys():
            # get tag that appears the mosts for given word
            model.loc[word] = max(possible_tag_for_words[word], key=possible_tag_for_words[word].get)
    return model


######### TASK b-ii #########

def compute_error_rate(model, test_set, known_words):
    # init error rate
    known_error_rate, unknown_error_rate = 0, 0

    # init counters
    word_counter, known_counter, unknown_counter = 0, 0, 0

    for sentence in test_set:
        for tup in sentence:
            word_counter += 1
            word, pred_tag = tup[0], tup[1]
            true_tag = model.loc[word][0]
            if word in known_words:
                known_counter += 1
                if (pred_tag != true_tag):
                    known_error_rate += 1
            else:
                unknown_counter += 1
                if (pred_tag != true_tag):
                    unknown_error_rate += 1

    error_rate = (known_error_rate + unknown_error_rate) / word_counter
    known_error_rate = (known_error_rate / known_counter)
    unknown_error_rate = (unknown_error_rate / unknown_counter)
    return error_rate, known_error_rate, unknown_error_rate


######### TASK b helpers #########

def get_data_from_train_set(train_set):
    """

    - used in main
    :return:     possible_tags_for_word: nested dict {word:{tag:frequency}}
                 tagged_sentences: list of list
                 tag_set: set of tags in train
                 known_words: set of words seen in train set

    """
    known_words = set()
    possible_tag_for_words = dict()
    tagged_sentences = []
    tag_set = ["START"]
    words_apperance = []
    for sentence in train_set:
        cur_sentence_tags = []
        cur_sentence = []

        for tup in sentence:
            word, tag = tup[0], tup[1]
            known_words.add(word)
            words_apperance.append(word)
            if "+" in tag or "-" in tag:
                tag = re.split(r'\+|\-', tag)[0]
            cur_sentence_tags.append(tag)

            tag_set.append(tag)
            if word not in possible_tag_for_words.keys():
                possible_tag_for_words[word] = dict()
                possible_tag_for_words[word][tag] = 1
            else:
                if tag in possible_tag_for_words[word].keys():
                    possible_tag_for_words[word][tag] += 1
                else:
                    possible_tag_for_words[word][tag] = 1
        tagged_sentences.append(cur_sentence_tags)
    return possible_tag_for_words, tagged_sentences, list(set(tag_set)), known_words, words_apperance


def print_errors(error_rate, known_error_rate, unknown_error_rate):
    print("known_error_rate = " + str(known_error_rate))
    print("unknown_error_rate = " + str(unknown_error_rate))
    print("error_rate = " + str(error_rate))


################################ TASK c ###############################
######### TASK c-i #########

def get_emission(words_set, tag_set, possible_tag_for_words):
    """
    init emission matrix from known words dict. for unseen words we define emission["NN"][word] = 1
    """
    words_set = list(set(words_set))
    emission = pd.DataFrame(data=np.zeros((len(tag_set), len(words_set))), index=tag_set,
                            columns=words_set)
    for word in words_set:
        if word in possible_tag_for_words.keys():
            tags_dict = possible_tag_for_words[word]
            for tag in tags_dict.keys():
                value = tags_dict[tag]
                emission.loc[tag][word] = value
        else:
            emission.loc["NN"][word] = 1

    # normalizing rows
    for tag in emission.index:
        if (np.sum(emission.loc[tag]) != 0):
            emission.loc[tag] /= np.sum(emission.loc[tag])

    return emission


def get_transition(tagged_sentences, tag_set):
    # row - reference tag - behinaten tag
    # tag_set = list(set(tag_set))
    words_arr = np.array(tag_set)
    transition = pd.DataFrame(data=np.zeros((words_arr.shape[0], words_arr.shape[0])), index=words_arr,
                              columns=words_arr)
    b_gram = dict()  # creating a nested dictionary. to hold biagram
    # create b_gram dict and fills it
    for doc in tagged_sentences:
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

    # fills transition with propabilites
    for word in b_gram.keys():
        cur_dict = b_gram[word]
        for next_word in cur_dict.keys():
            value = cur_dict[next_word]
            # transition[next_word][word] = value #TODO  why its working?
            transition.loc[word][next_word] = value  # TODO  why its working?
        transition.loc[word] /= np.sum(list(cur_dict.values()))

    return transition


######### TASK c2 - helpers #########

def split_test_set(test_set):
    word_sentences = []
    test_tags = []
    only_tags = set()
    for sentence in test_set:
        cur_word_sentence = ["START"]
        cur_true_tags = ["START"]
        for tup in sentence:
            word = tup[0]
            tag = tup[1]
            only_tags.add(tag)
            if "+" in tag or "-" in tag:
                tag = re.split(r'\+|\-', tag)[0]
            cur_word_sentence.append(word)
            cur_true_tags.append(tag)
        word_sentences.append(cur_word_sentence)
        test_tags.append(cur_true_tags)
    return word_sentences, test_tags,only_tags


# init matrices


######### TASK c-ii #########
def viterbi(sentence, transition, emission, tag_set):
    # tag_set =list(set(tag_set))
    # dynamic_table = pd.DataFrame(data=np.zeros((len(tag_set), len(sentence))), index=tag_set,columns=sentence)
    # indx_table = pd.DataFrame(data=np.zeros((len(tag_set), len(sentence))), index=tag_set,
    #                              columns=sentence)
    # init table
    prob_table = np.zeros((len(tag_set), len(sentence)))
    idx_table = np.zeros((len(tag_set), len(sentence)))
    prob_table[:, 0] = int(1)

    for col in range(1, len(sentence)):  # TODO right way?
        word = sentence[col]
        for row, tag in enumerate(tag_set):
            # get probs
            emission_prop = emission.loc[tag][word]
            transition_prop = np.array(transition)[:, row]
            prev_col_arr = prob_table[:, col - 1]

            maxvalue = np.max(prev_col_arr * transition_prop * emission_prop)
            # print(maxvalue)
            prob_table[row, col] = maxvalue
            idx_table[row, col] = np.argmax(prev_col_arr * transition_prop * emission_prop)

    row_idx = int(np.argmax(prob_table[:, -1]))
    if row_idx == 0:
        return ["NN"] * len(sentence)
    # run on sentence -- run on
    # extract path
    tags = []
    # print("row_idx " + str(row_idx) + "max " + str(np.max(prob_table[:,-1])))

    for col in range(prob_table.shape[1] - 1, 0, -1):  # run over columns backwards

        tag = tag_set[row_idx]
        tags.append(tag)
        row_idx = int(idx_table[row_idx, col])

    return ["START"] + tags[::-1]


######### TASK c-iii #########
# def compute_error_rate_vertibi(test_set, known_words, emission, transition, tag_set):
#     # init error rate
#     known_error_rate, unknown_error_rate = 0, 0
#
#     # init counters
#     word_counter, known_counter, unknown_counter = 0, 0, 0
#
#     word_sentences, test_tags = split_test_set(test_set)
#     for i, sentence in enumerate(word_sentences):
#         pred_tags = viterbi(sentence, transition, emission, tag_set)
#         true_tags = test_tags[i]  # NOTE:tags for current sentence
#         # print( f"{pred_tags},{true_tags},{sentence}")
#         for j, word in enumerate(sentence):
#             word_counter += 1
#             if word in known_words:
#                 known_counter += 1
#                 assert len(pred_tags) == len(true_tags), f"{pred_tags},{true_tags},{sentence}"
#                 if pred_tags[j] != true_tags[j]:
#                     known_error_rate += 1
#             else:
#                 unknown_counter += 1
#                 if pred_tags[j] != true_tags[j]:
#                     unknown_error_rate += 1
#
#     error_rate = (known_error_rate + unknown_error_rate) / word_counter
#     known_error_rate = (known_error_rate / known_counter)
#     unknown_error_rate = (unknown_error_rate / unknown_counter)
#     return error_rate, known_error_rate, unknown_error_rate


#
def compute_error_rate_vertibi(test_set, known_words, emission,transition,tag_set):
    # init error rate
    known_error_rate, unknown_error_rate = 0, 0

    # init counters
    word_counter, known_counter, unknown_counter = 0, 0, 0

    # init confusion_matrix size (tag_set*tag_set)
    #NOTE does tag_set conatin all tags including categ/pred/train
    word_sentences, test_tags,only_tags = split_test_set(test_set)
    total_tags = only_tags.union(set(tag_set))
    conf_matrix = pd.DataFrame(data=np.zeros((len(total_tags),len(total_tags))), index=list(total_tags),columns=list(total_tags))
    for i, sentence in enumerate(word_sentences):
        pred_tags = viterbi(sentence, transition, emission, tag_set)
        true_tags = test_tags[i]  # NOTE:tags for current sentence

        # print( f"{pred_tags},{true_tags},{sentence}")
        for j, word in enumerate(sentence):
            word_counter += 1
            if word in known_words:
                known_counter += 1
                assert len(pred_tags) == len(true_tags), f"{pred_tags},{true_tags},{sentence}"
                if pred_tags[j] != true_tags[j]:
                    known_error_rate += 1
                    conf_matrix.loc[true_tags[j]][pred_tags[j]] +=1
            else:
                unknown_counter += 1
                if pred_tags[j] != true_tags[j]:
                    unknown_error_rate += 1
                    conf_matrix.loc[true_tags[j]][pred_tags[j]] += 1


    error_rate = (known_error_rate + unknown_error_rate) / word_counter
    known_error_rate = (known_error_rate / known_counter)
    unknown_error_rate = (unknown_error_rate / unknown_counter)
    return error_rate, known_error_rate, unknown_error_rate, conf_matrix


################################ TASK d ###############################
######### TASK d-i #########

def get_smoothy_emission(words_set, tag_set, possible_tag_for_words):
    """
    init emission matrix from known words dict. for unseen words we define emission["NN"][word] = 1
    """
    words_set = list(set(words_set))
    emission = pd.DataFrame(data=np.zeros((len(tag_set), len(words_set))), index=tag_set,
                            columns=words_set)
    for word in words_set:
        if word in possible_tag_for_words.keys():
            tags_dict = possible_tag_for_words[word]
            for tag in tags_dict.keys():
                value = tags_dict[tag]
                emission.loc[tag][word] = value
        else:
            emission.loc["NN"][word] = 1
    # smooth (add one)
    emission += 1
    # normalizing rows
    for tag in emission.index:
        emission.loc[tag] /= np.sum(emission.loc[tag])

    return emission


################################ TASK e ###############################
######### TASK e-i #########

def get_psudo_words(words_apperance, words_set, known_words, possible_tag_for_words):
    low_freq(known_words, possible_tag_for_words, words_apperance)
    # low_freq_tag_dict = dict()
    # for word in low_freq_words:
    #
    #     for tag in possible_tag_for_words[word].keys():
    #         if tag in low_freq_tag_dict.keys():
    #             low_freq_tag_dict[tag] += possible_tag_for_words[word][tag]
    #         else:
    #             low_freq_tag_dict[tag] = possible_tag_for_words[word][tag]
    # best_tag = max(low_freq_tag_dict, key=low_freq_tag_dict.get)
    # for word in low_freq_words:
    #     value = sum(possible_tag_for_words[word].values())
    #     temp_dict = dict()
    #     temp_dict[best_tag] = value
    #     possible_tag_for_words[word] = dict([(best_tag,value)]) # create nested dict {best_tag: # of instances of the word in corpus

    category_1 = dict()
    category_2 = dict()
    category_3 = dict()
    for word in known_words:
        if word.isnumeric():  # category 1
            for tag in possible_tag_for_words[word]:
                if tag in category_1:
                    category_1[tag] += possible_tag_for_words[word][tag]
                else:
                    category_1[tag] = possible_tag_for_words[word][tag]
        elif word[0].isupper():  # category 2
            for tag in possible_tag_for_words[word]:
                if tag in category_2:
                    category_2[tag] += possible_tag_for_words[word][tag]
                else:
                    category_2[tag] = possible_tag_for_words[word][tag]
        elif word.isupper():  # category 3
            for tag in possible_tag_for_words[word]:
                if tag in category_3:
                    category_3[tag] += possible_tag_for_words[word][tag]
                else:
                    category_3[tag] = possible_tag_for_words[word][tag]
    categories_dict = dict()
    categories_dict["NUM"] = max(category_1, key=category_1.get)  # NOTE best tag for this specific category
    categories_dict["FIRSTCAP"] = max(category_2, key=category_2.get)
    categories_dict["ALLCAP"] = max(category_3, key=category_3.get)
    for word in possible_tag_for_words.keys():
        value = sum(possible_tag_for_words[word].values())
        if word.isnumeric():  # category 1
            possible_tag_for_words[word] = {categories_dict["NUM"]: value}
        elif word[0].isupper():  # category 2
            possible_tag_for_words[word] = {categories_dict["FIRSTCAP"]: value}
        elif word.isupper():  # category 3
            possible_tag_for_words[word] = {categories_dict["ALLCAP"]: value}

    return possible_tag_for_words


def low_freq(known_words, possible_tag_for_words, words_apperance):
    low_freq_words = set()
    category_1 = dict()
    category_2 = dict()
    category_3 = dict()
    low_freq_tag_dict = dict()
    low = set()
    for word in known_words:
        if words_apperance.count(word) < 2:  # check low freq
            # low_freq_words.add(word)
            if word.isnumeric():  # category 1
                low_freq_words.add(word)
                for tag in possible_tag_for_words[word]:
                    if tag in category_1:
                        category_1[tag] += possible_tag_for_words[word][tag]
                    else:
                        category_1[tag] = possible_tag_for_words[word][tag]
            elif word[0].isupper():  # category 2
                low_freq_words.add(word)
                for tag in possible_tag_for_words[word]:
                    if tag in category_2:
                        category_2[tag] += possible_tag_for_words[word][tag]
                    else:
                        category_2[tag] = possible_tag_for_words[word][tag]
            elif word.isupper():  # category 3
                low_freq_words.add(word)
                for tag in possible_tag_for_words[word]:
                    if tag in category_3:
                        category_3[tag] += possible_tag_for_words[word][tag]
                    else:
                        category_3[tag] = possible_tag_for_words[word][tag]
            else:
                for tag in possible_tag_for_words[word].keys():
                    low.add(word)
                    if tag in low_freq_tag_dict.keys():
                        low_freq_tag_dict[tag] += possible_tag_for_words[word][tag]
                    else:
                        low_freq_tag_dict[tag] = possible_tag_for_words[word][tag]
    best_tag = max(low_freq_tag_dict, key=low_freq_tag_dict.get)
    for word in low:
        value = sum(possible_tag_for_words[word].values())
        temp_dict = dict()
        temp_dict[best_tag] = value
        possible_tag_for_words[word] = dict(
            [(best_tag, value)])  # create nested dict {best_tag: # of instances of the word in corpus
    categories_dict = dict()
    categories_dict["NUM"] = max(category_1, key=category_1.get)  # NOTE best tag for this specific category
    categories_dict["FIRSTCAP"] = max(category_2, key=category_2.get)
    categories_dict["ALLCAP"] = max(category_3, key=category_3.get)
    for word in low_freq_words:
        value = sum(possible_tag_for_words[word].values())
        if word.isnumeric():  # category 1
            possible_tag_for_words[word] = {categories_dict["NUM"]: value}
        elif word[0].isupper():  # category 2
            possible_tag_for_words[word] = {categories_dict["FIRSTCAP"]: value}
        elif word.isupper():  # category 3
            possible_tag_for_words[word] = {categories_dict["ALLCAP"]: value}


######### TASK e-2 #########


def categorise_viterbi(sentence, transition, emission, tag_set, categories_dict):
    # init table
    prob_table = np.zeros((len(tag_set), len(sentence)))
    idx_table = np.zeros((len(tag_set), len(sentence)))
    prob_table[:, 0] = int(1)

    for col in range(1, len(sentence)):  # TODO right way?
        word = sentence[col]
        for row, tag in enumerate(tag_set):
            # get probs
            emission_prop = emission.loc[tag][word]
            transition_prop = np.array(transition)[:, row]
            prev_col_arr = prob_table[:, col - 1]

            maxvalue = np.max(prev_col_arr * transition_prop * emission_prop)
            # print(maxvalue)
            prob_table[row, col] = maxvalue
            idx_table[row, col] = np.argmax(prev_col_arr * transition_prop * emission_prop)

    row_idx = int(np.argmax(prob_table[:, -1]))
    if row_idx == 0:
        return ["NN"] * len(sentence)
    # extract path
    tags = []
    for col in range(prob_table.shape[1] - 1, 0, -1):  # run over columns backwards

        tag = tag_set[row_idx]
        tags.append(tag)
        row_idx = int(idx_table[row_idx, col])

    return ["START"] + tags[::-1]


if __name__ == '__main__':
    # A
    train_set, test_set, data = load_data()

    # mid
    possible_tag_for_words, tagged_sentences, tag_set, known_words, words_apperance = get_data_from_train_set(train_set)
    words_set = set(np.array(brown.words()))  # all words in data

    # B-i
    model = init_model(possible_tag_for_words, words_set)
    # B-ii
    print("B-ii")
    error_rate, known_error_rate, unknown_error_rate = compute_error_rate(model, test_set, known_words)
    print_errors(error_rate, known_error_rate, unknown_error_rate)
    #
    # # C-i
    emission = get_emission(words_set, tag_set, possible_tag_for_words)
    transition = get_transition(tagged_sentences, tag_set)
    #
    # # C-iii
    print("C-iii")

    error_rate, known_error_rate, unknown_error_rate, confusion_matrix1 = compute_error_rate_vertibi(test_set, known_words, emission,
                                                                   transition, tag_set)
    print_errors(error_rate, known_error_rate, unknown_error_rate)
    fig = px.imshow(confusion_matrix1)
    fig.show()
    print()
    # # D-i
    print("D-ii")

    smoothed_emission = get_smoothy_emission(words_set, tag_set, possible_tag_for_words)
    error_rate, known_error_rate, unknown_error_rate, confusion_matrix2 = compute_error_rate_vertibi(test_set, known_words,
                                                                                  smoothed_emission, transition,
                                                                                  tag_set)
    print_errors(error_rate, known_error_rate, unknown_error_rate)
    fig = px.imshow(confusion_matrix2)
    fig.show()
    # E-i
    possible_tag_for_words = get_psudo_words(words_apperance, words_set, known_words, possible_tag_for_words)

    # E-ii
    smoothed_emission = get_smoothy_emission(words_set, tag_set, possible_tag_for_words)
    transition = get_transition(tagged_sentences, tag_set)
    print("E-ii")
    error_rate, known_error_rate, unknown_error_rate, confusion_matrix3 = compute_error_rate_vertibi(test_set, known_words, smoothed_emission,
                                                                                  transition, tag_set)
    print_errors(error_rate, known_error_rate, unknown_error_rate)
    fig = px.imshow(confusion_matrix3)
    fig.show()
