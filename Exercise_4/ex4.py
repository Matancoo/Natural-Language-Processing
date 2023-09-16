################################ Imports ######################################
from sklearn.utils import shuffle

import nltk
import numpy as np, pandas as pd
from sklearn.base import BaseEstimator
import plotly.express as px
from collections import Counter
import matplotlib.pyplot as plt

from Chu_Liu_Edmonds_algorithm import max_spanning_arborescence_nx
from nltk.corpus import dependency_treebank
from sklearn.model_selection import train_test_split

WORD = "word"

nltk.download('dependency_treebank')
import numpy as np
import pickle #to save and load data


################################ Constants ####################################

NODE_TAG = "tag"
NODE_ADDRESS = "address"


################################ Model Class ####################################

class Arc:
    """
    class for a given edge in sentence
    """

    def __init__(self, head, tail, score):
        self.head = head
        self.tail = tail
        self.weight = 0  # initial score

    def set_score(self, weight):
        # assert weight >= 0
        self.weight = -weight

    def get_edge(self):
        return (self.head, self.tail)

#TODO: when should we separate classes: ????

# class DataLoader(self,test_size = 0.1):
#     def __init__(self):
#         self.data = dependency_treebank.parsed_sents()
#         self.train, self.test = train_test_split(self.data, test_size=)
#         self.words = set(np.array(dependency_treebank.words()))  # save all words from corpus
#         self.tags = self.get_tags()
#         self.words_length = len(self.words)
#         self.tags_length = len(self.tags)
#         self.word_dict = dict([(word, index) for index, word in enumerate(self.words)])  # get index of word
#         self.tag_dict = dict(
#             [(tag, index) for index, tag in enumerate(self.tags)])  # get index of tag
#         self.feature_length = np.square(self.words_length) + np.square(self.tags_length)
#



class Model:

    def __init__(self,lr = 1,epoch = 2):
        self.load_data()
        self.words = set(np.array(dependency_treebank.words()))  # save all words from corpus
        # self.tags = set(dict(dependency_treebank.tagged_words()).values())
        tag = [val for val in dependency_treebank.tagged_sents()]
        self.tags = self.get_tags()
        # self.tags.add("TOP")
        self.words_length = len(self.words)
        self.tags_length = len(self.tags)
        self.word_dict = dict([(word, index) for index, word in enumerate(self.words)])  # get index of word
        self.tag_dict = dict(
            [(tag, index) for index, tag in enumerate(self.tags)])  # get index of tag
        self.feature_length = np.square(self.words_length) + np.square(self.tags_length)
        self.cur_weight = np.zeros(self.feature_length)
        # perceptron variables
        self.epoch = epoch
        # self.cumulative_weights = np.zeros(self.feature_length)
        self.lr = lr
        self.train_accuracy = []

    ########### init helper ###########

    def get_tags(self):
        tags = set()
        for sentence in dependency_treebank.tagged_sents():
            for tup in sentence:
                tags.add(tup[1])
        return tags

    def load_model(self, weights,train_accuracy):
        self.cur_weight = weights
        self.train_accuracy = train_accuracy


    ########### Task 1 ###########
    def load_data(self):
        self.data = dependency_treebank.parsed_sents()
        self.train, self.test = train_test_split(self.data, test_size=0.1)

    ########### Task 2 ###########

    def feature_function(self, edge, sentence_tag_dict):
        # create zero feature vector
        # feature_vec = np.zeros(self.feature_length)

        word1, word2 = edge[0], edge[1]  # get words of edge

        tag1, tag2 = sentence_tag_dict[word1], sentence_tag_dict[word2]



        word1_index, tag1_index = self.word_dict[word1], self.tag_dict[tag1]  # rows
        word2_index, tag2_index = self.word_dict[word2], self.tag_dict[tag2]  # cols

        # fill feature vector
        edge_word_index = (word1_index * self.words_length) + word2_index
        edge_tag_index = np.square(self.words_length) + (
                tag1_index * self.tags_length) + tag2_index  # np.square(self.words_length) = V^2
        # feature_vec[edge_word_index] = 1
        # feature_vec[edge_tag_index] = 1

        return (edge_word_index, edge_tag_index)

    ########### Task 3 - helpers ###########

    def get_sentence_tag_dict_from_tree(self, tree) -> dict:
        sentence_tag_dict = dict()
        for i in range(1, len(tree.nodes)):  # skip first node which node["tag"] = "TOP"
            word = tree.nodes[i][WORD]
            sentence_tag_dict[word] = tree.nodes[i][NODE_TAG]
        return sentence_tag_dict

    def get_edges_from_word_set(self, words_of_tree):
        return [Arc(w1, w2, 0) for w1 in words_of_tree for w2 in words_of_tree if w1 != w2]

    def get_feature_vec_of_tree(self, tree, sentence_tag_dict, head, ind_list = list()):
        childrens_indexs = list(head["deps"].values())

        if not childrens_indexs:
            return
        for tail_index in childrens_indexs[0]:  # run on all tails of head
            tail_node = tree.nodes[tail_index]
            if head[WORD] is None:
                self.get_feature_vec_of_tree(tree, sentence_tag_dict, tail_node,ind_list)
            else:
                edge = (head[WORD], tail_node[WORD])
                edge_word_index, edge_tag_index = self.feature_function(edge, sentence_tag_dict)
                ind_list.append(edge_word_index)
                ind_list.append(edge_tag_index)

                self.get_feature_vec_of_tree(tree, sentence_tag_dict, tail_node, ind_list)

        return ind_list

    def get_feature_vec_of_chi_liu_tree(self, chi_liu_tree, sentence_tag_dict, ind_list= list()):

        # head = [word for word in chi_liu_tree.keys() if chi_liu_tree[word].head() == "None"][0]
        # feature_vec_sum = np.zeros(self.feature_length)
        for tail in chi_liu_tree.keys():
            head = chi_liu_tree[tail].head
            if head == "None":
                continue
            edge = (head, tail)
            edge_word_index, edge_tag_index = self.feature_function(edge, sentence_tag_dict)
            ind_list.append(edge_word_index)
            ind_list.append(edge_tag_index)
        return ind_list

    ########### Task 3 - Perceptron ###########

    def fit(self):
        cur_weight = np.zeros(self.feature_length).astype(np.float32)
        cumulative_weights = np.zeros(self.feature_length).astype(np.float32)
        print("fit with train length " + str(len(self.train)))
        for epoch in range(self.epoch):
            print("epoch" + str(epoch))
            i = 0

            for true_tree in shuffle(self.train):  # shufle train data to train better
                print(i)
                i += 1

                sentence_tag_dict = self.get_sentence_tag_dict_from_tree(true_tree)  # sentence_word -> sentence_tag
                words_of_tree = sentence_tag_dict.keys()
                all_possible_arcs = self.get_edges_from_word_set(words_of_tree)  # create full graph

                # build full tree ang score it
                for arc in all_possible_arcs:
                    edge = arc.get_edge()  # (w_i , w_j)
                    edge_word_index, edge_tag_index = self.feature_function(edge, sentence_tag_dict)
                    score = cur_weight[edge_word_index] + cur_weight[edge_tag_index]
                    arc.set_score(score)

                all_possible_arcs.append(
                    Arc(head="None", tail=true_tree.root["word"], score=0))  # connect ("None","ROOT")
                # sentence_tag_dict["None"] = "TOP"

                chi_liu_tree = max_spanning_arborescence_nx(arcs=all_possible_arcs, sink=0)

                true_tree_feature = np.array(self.get_feature_vec_of_tree(true_tree, sentence_tag_dict, true_tree.nodes[0]))


                chu_liu_tree_feature = np.array(self.get_feature_vec_of_chi_liu_tree(chi_liu_tree, sentence_tag_dict))


                cur_weight[true_tree_feature] +=1
                cur_weight[chu_liu_tree_feature] -=1


                # calculate accuracy
                total_arcs = len(true_tree_feature)
                correct_arcs = np.sum([1 for idx in true_tree_feature if idx in chu_liu_tree_feature])
                self.train_accuracy.append(correct_arcs / total_arcs)
                cumulative_weights += cur_weight

        print("fit done")
        self.cur_weight = cur_weight
        return np.sum(cumulative_weights) / (self.epoch * len(self.train))  # todo make sure its ok

    ########### Task 4  ###########
    def evaluate(self):
        # how extract test sentence
        self.test_accuracy = []  # init
        print("evaluate test with length " + str(len(self.test)))
        i = 0
        for true_tree in self.test:
            print(i)
            i += 1
            sentence_tag_dict = self.get_sentence_tag_dict_from_tree(true_tree)  # sentence_word -> sentence_tag
            words_of_tree = sentence_tag_dict.keys()
            all_possible_arcs = self.get_edges_from_word_set(words_of_tree)

            # build full tree ang score it
            for arc in all_possible_arcs:
                edge = arc.get_edge()  # (w_i , w_j)
                edge_word_index, edge_tag_index = self.feature_function(edge, sentence_tag_dict)
                score = self.cur_weight[edge_word_index] + self.cur_weight[edge_tag_index]
                arc.set_score(score)

            chi_liu_tree = max_spanning_arborescence_nx(arcs=all_possible_arcs, sink=0)
            true_tree_feature = self.get_feature_vec_of_tree(true_tree, sentence_tag_dict, true_tree.nodes[0])
            chu_liu_tree_feature = self.get_feature_vec_of_chi_liu_tree(chi_liu_tree, sentence_tag_dict)

            # calculate accuracy
            total_arcs = len(true_tree_feature)
            correct_arcs = np.sum([1 for idx in true_tree_feature if idx in chu_liu_tree_feature])
            self.test_accuracy.append(correct_arcs / total_arcs)
        print("evaluate done")

        accuracy = np.average(self.test_accuracy)
        return accuracy

    ########### Task 1.3 ###########
    def plot_test_accuracy(self, accuracys , name):
        print("ploting")
        # with lstm we run with different color
        # x_train = np.arange(len(self.train_losses))
        x_test = np.arange(len(accuracys))

        # plt.plot(x_train, self.train_losses, 'b', label='Train loss')
        plt.plot(x_test, accuracys, 'y', label='Accuracy')
        plt.xlabel("Iteration")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title(name)
        plt.show()


    ########### Task 1.4 ###########

    ########### Task 1.5 ###########


################################ helpers ###############################
    # save it
def save_model(weights, train_accuracy):
    with open('modelWeights.pkl', 'wb') as file:
        pickle.dump((weights,train_accuracy), file)

    # load it
def load_model():
    with open('modelWeights.pkl', 'rb') as file:
        weights,train_accuracy = pickle.load(file)
        return weights,train_accuracy

if __name__ == '__main__':
    model = Model()

    # fit and save wights
    model.fit()
    save_model(model.cur_weight, model.train_accuracy)

    # load weights - no need to fit
    # weights,train_accuracy = load_model()
    # model.load_model(weights, train_accuracy)


    accuracy = model.evaluate()
    print(accuracy)
    model.plot_test_accuracy(model.train_accuracy, "Accuracy over train set")
    model.plot_test_accuracy(model.test_accuracy, "Accuracy over test set")

