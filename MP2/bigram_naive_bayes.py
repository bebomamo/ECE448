# bigram_naive_bayes.py
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
from tqdm import tqdm
from collections import Counter
from math import log


'''
utils for printing values
'''
def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
    print(f"Positive prior: {pos_prior}")

def print_values_bigram(unigram_laplace, bigram_laplace, bigram_lambda, pos_prior):
    print(f"Unigram Laplace: {unigram_laplace}")
    print(f"Bigram Laplace: {bigram_laplace}")
    print(f"Bigram Lambda: {bigram_lambda}")
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
Main function for training and predicting with the bigram mixture model.
    You can modify the default values for the Laplace smoothing parameters, model-mixture lambda parameter, and the prior for the positive label.
    Notice that we may pass in specific values for these parameters during our testing.
"""
def bigram_bayes(train_set, train_labels, dev_set, unigram_laplace=.001, bigram_laplace=.005, bigram_lambda=.5, pos_prior=0.8, silently=False):
    print_values_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior)
    
    # Making the training dictionaries (positive and negative)
    unigram_negative_freq_map = dict()
    unigram_positive_freq_map = dict()
    bigram_negative_freq_map = dict()
    bigram_positive_freq_map = dict()
    unigram_negative_total_n = 0
    unigram_positive_total_n = 0
    bigram_negative_total_n = 0
    bigram_positive_total_n = 0
    # Unigram training session
    [unigram_positive_total_n, unigram_negative_total_n] = unigram_train(train_set, train_labels, unigram_positive_freq_map, unigram_negative_freq_map)
    # Bigram training session
    [bigram_positive_total_n, bigram_negative_total_n] = bigram_train(train_set, train_labels, bigram_positive_freq_map, bigram_negative_freq_map)
    # Log-based frequency calculations
    [positive_words_types, negative_words_types, positive_key_types, negative_key_types] = frequency_calculations(unigram_positive_freq_map, unigram_negative_freq_map, bigram_positive_freq_map, bigram_negative_freq_map, unigram_laplace, bigram_laplace, 
                                                                                                                unigram_positive_total_n, unigram_negative_total_n, bigram_positive_total_n, bigram_negative_total_n)
    # Reading the dev_set and doing the naive bayes
    # P(positive|words) = P(positive(prior)) + Sum{P(word|type=positive)}
    neg_prior = 1-pos_prior
    yhats = []
    unigram_negative_pmass = []
    unigram_positive_pmass = []
    bigram_negative_pmass = []
    bigram_positive_pmass = []
    for doc in tqdm(dev_set, disable=silently): #initialization of yhats
        yhats.append(-1)
    
    for review in dev_set:
        P_positive_review = log(pos_prior)
        P_negative_review = log(neg_prior)
        for word in review:
            # Positive Summation step
            if word in unigram_positive_freq_map:
                P_positive_review += unigram_positive_freq_map[word]
            else:
                P_positive_review += log(unigram_laplace / (unigram_positive_total_n + unigram_laplace*(positive_words_types+1)))
            #Negative Summation step
            if word in unigram_negative_freq_map:
                P_negative_review += unigram_negative_freq_map[word]
            else:
                P_negative_review += log(unigram_laplace / (unigram_negative_total_n + unigram_laplace*(negative_words_types+1)))
        # Store probability masses for bigram-unigram mixture evaluation
        unigram_positive_pmass.append(P_positive_review)
        unigram_negative_pmass.append(P_negative_review)

    for review in dev_set:
        P_positive_review = log(pos_prior)
        P_negative_review = log(neg_prior)
        length = len(review)
        for i in range(0, length-1):
            key = (review[i],review[i+1])
            # Positive Summation
            if key in bigram_positive_freq_map:
                P_positive_review += bigram_positive_freq_map[key]
                # print(i)
            else:
                P_positive_review += log(bigram_laplace / (bigram_positive_total_n + bigram_laplace*(positive_key_types+1)))
            #Negative Summation step
            if key in bigram_negative_freq_map:
                P_negative_review += bigram_negative_freq_map[key]
            else:
                P_negative_review += log(bigram_laplace / (bigram_negative_total_n + bigram_laplace*(negative_key_types+1)))
        # Store probability masses for bigram-unigram mixture evaluation
        bigram_positive_pmass.append(P_positive_review)
        bigram_negative_pmass.append(P_negative_review)

    # combining the bigram and unigram models
    for index in range(0, len(dev_set)):
        P_pos = (1-bigram_lambda)*unigram_positive_pmass[index] + bigram_lambda*bigram_positive_pmass[index]
        P_neg = (1-bigram_lambda)*unigram_negative_pmass[index] + bigram_lambda*bigram_negative_pmass[index]
        yhats[index] = 1 if (P_pos > P_neg) else 0
        # yhats[index] = 1 if (unigram_positive_pmass[index] > unigram_negative_pmass[index]) else 0
        
    return yhats

# Trains the Unigram model, returns the positive and negative token count
def unigram_train(train_set, train_labels, unigram_positive_freq_map, unigram_negative_freq_map):
    train_labels_index = 0
    unigram_positive_tokens = 0
    unigram_negative_tokens = 0
    for list in train_set:
        for word in list:
            if(train_labels[train_labels_index] == 1): #i.e. positive review
                if word in unigram_positive_freq_map:
                    unigram_positive_freq_map[word] += 1
                else:
                    unigram_positive_freq_map[word] = 1
                unigram_positive_tokens += 1
            else: #i.e. negative review
                if word in unigram_negative_freq_map:
                    unigram_negative_freq_map[word] += 1
                else:
                    unigram_negative_freq_map[word] = 1
                unigram_negative_tokens += 1
        train_labels_index += 1
    return [unigram_positive_tokens, unigram_negative_tokens]

# Trains the Bigram model, returns the positive and negative token count
def bigram_train(train_set, train_labels, bigram_positive_freq_map, bigram_negative_freq_map):
    train_labels_index = 0
    bigram_positive_tokens = 0
    bigram_negative_tokens = 0
    for list in train_set:
        length = len(list)
        for i in range(0,length-1):
            key = (list[i],list[i+1])
            if(train_labels[train_labels_index] == 1): #i.e. positive review
                if key in bigram_positive_freq_map:
                    bigram_positive_freq_map[key] += 1 
                else:
                    bigram_positive_freq_map[key] = 1
                bigram_positive_tokens += 1
            else: #i.e. negative review
                if key in bigram_negative_freq_map:
                    bigram_negative_freq_map[key] += 1
                else:
                    bigram_negative_freq_map[key] = 1
                bigram_negative_tokens += 1
        train_labels_index += 1
    return [bigram_positive_tokens, bigram_negative_tokens]

def frequency_calculations(unigram_positive_freq_map, unigram_negative_freq_map, bigram_positive_freq_map, bigram_negative_freq_map, unigram_laplace, bigram_laplace, 
                           unigram_positive_total_n, unigram_negative_total_n, bigram_positive_total_n, bigram_negative_total_n):
    # Unigram Calculating frequency probabilities
    positive_words_types = len(unigram_positive_freq_map)
    for key, value in unigram_positive_freq_map.items():
        unigram_positive_freq_map[key] = log((value+unigram_laplace)/(unigram_positive_total_n + unigram_laplace*(positive_words_types+1)))

    negative_words_types = len(unigram_negative_freq_map)
    for key, value in unigram_negative_freq_map.items():
        unigram_negative_freq_map[key] = log((value+unigram_laplace)/(unigram_negative_total_n + unigram_laplace*(negative_words_types+1)))

    # Bigram Calculating frequency probabilities
    positive_key_types = len(bigram_positive_freq_map) 
    for key, value in bigram_positive_freq_map.items():
        bigram_positive_freq_map[key] = log((value+bigram_laplace)/(bigram_positive_total_n + bigram_laplace*(positive_key_types+1)))

    negative_key_types = len(bigram_negative_freq_map)
    for key, value in bigram_negative_freq_map.items():
        bigram_negative_freq_map[key] = log((value+bigram_laplace)/(bigram_negative_total_n + bigram_laplace*(negative_key_types+1)))

    return [positive_words_types, negative_words_types, positive_key_types, negative_key_types]