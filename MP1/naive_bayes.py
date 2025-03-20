# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Last Modified 8/23/2023


"""
This is the main code for this MP.
You only need (and should) modify code within this file.
Original staff versions of all other files will be used by the autograder
so be careful to not modify anything else.
"""


import reader
import math
from math import log
from tqdm import tqdm
from collections import Counter


'''
util for printing values
'''
def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
    print(f"Positive prior: {pos_prior}")

"""
load_data loads the input data by calling the provided utility.
You can adjust default values for stemming and lowercase, when we haven't passed in specific values,
to potentially improve performance.
"""
def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming: {stemming}")
    print(f"Lowercase: {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels


"""
Main function for training and predicting with naive bayes.
    You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
    Notice that we may pass in specific values for these parameters during our testing.
"""
def naive_bayes(train_set, train_labels, dev_set, laplace=9.25, pos_prior=0.8, silently=False):
    print_values(laplace,pos_prior)

    # Making the training dictionaries (positive and negative)
    negative_freq_map = dict()
    positive_freq_map = dict()
    negative_total_n = 0
    positive_total_n = 0
    train_labels_index = 0
    for list in train_set:
        for word in list:
            if(train_labels[train_labels_index] == 1): #i.e. positive review
                if word in positive_freq_map:
                    positive_freq_map[word] += 1
                else:
                    positive_freq_map[word] = 1
                positive_total_n += 1
            else: #i.e. negative review
                if word in negative_freq_map:
                    negative_freq_map[word] += 1
                else:
                    negative_freq_map[word] = 1
                negative_total_n += 1
        train_labels_index += 1

    # Calculating frequency probabilities
    positive_words_types = len(positive_freq_map)
    for key, value in positive_freq_map.items():
        positive_freq_map[key] = log((value+laplace)/(positive_total_n + laplace*(positive_words_types+1)))

    negative_words_types = len(negative_freq_map)
    for key, value in negative_freq_map.items():
        negative_freq_map[key] = log((value+laplace)/(negative_total_n + laplace*(negative_words_types+1)))

    # Reading the dev_set and doing the naive bayes
    # P(positive|words) = P(positive(prior)) + Sum{P(word|type=positive)}
    neg_prior = 1-pos_prior
    index = 0
    yhats = []
    for doc in tqdm(dev_set, disable=silently): #initialization of yhats
        yhats.append(-1)
    
    for review in dev_set:
        P_positive_review = log(pos_prior)
        P_negative_review = log(neg_prior)
        for word in review:
            # Positive Summation step
            if word in positive_freq_map:
                P_positive_review += positive_freq_map[word]
            else:
                # P(UNK|REVIEWS) = a / n + a(V+1)
                positive_total_n += 1
                positive_words_types += 1
                positive_freq_map[word] = log(laplace / (positive_total_n + laplace*(positive_words_types+1)))
                P_positive_review += positive_freq_map[word]
            #Negative Summation step
            if word in negative_freq_map:
                P_negative_review += negative_freq_map[word]
            else:
                negative_total_n += 1
                negative_words_types += 1
                negative_freq_map[word] = log(laplace / (negative_total_n + laplace*(negative_words_types+1)))
                P_negative_review += negative_freq_map[word]
        # So now we make the MAP decision based on probabilities (1 is positive, 0 is negative)
        yhats[index] = 1 if (P_positive_review > P_negative_review) else 0
        index += 1
        
    return yhats
