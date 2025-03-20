"""
Part 3: Here you should improve viterbi to use better laplace smoothing for unseen words
This should do better than baseline and your first implementation of viterbi, especially on unseen words
Most of the code in this file is the same as that in viterbi_1.py
"""
import math
from collections import defaultdict, Counter
from math import log

epsilon_for_pt = 1e-5
emit_epsilon = 1e-5 

def hapax_handler(words, hapax):
    # Filtering out not hapax words and creating hapax tag probabilities
    words.sort()
    hapax_words = []
    length = len(words)
    i = 0
    while i < length-1: #python is so stupid for not allowing manual in-loop increment
        j = i
        if j + 1 < length:
            j += 1
            if words[i] != words[j]:
                hapax_words.append(words[i])
            else:
                while (j < length and str(words[i]) == str(words[j])):
                    j = j + 1
                i = j - 1
        i += 1
        
    sorted_hapax = {}
    hapax_tag_probs = {}
    total_hapax_tags = 0
    for word in hapax_words: #recover the tags for the hapax words
        sorted_hapax[word] = hapax[word]
    for word,tag in sorted_hapax.items(): #count tag frequencies amongst hapax' 
        total_hapax_tags += 1
        if tag in hapax_tag_probs:
            hapax_tag_probs[tag] += 1
        else:
            hapax_tag_probs[tag] = 1
    for tag in hapax_tag_probs: #divide by the total_hapax_tags
        hapax_tag_probs[tag] = hapax_tag_probs[tag]/total_hapax_tags
    hapax_tag_probs["unseen_tag"] = 1 / total_hapax_tags

    return hapax_tag_probs

def training(sentences):
    """
    Computes initial tags, emission words and transition tag-to-tag probabilities
    :param sentences:
    :return: intitial tag probs, emission words given tag probs, transition of tags to tags probs
    """
    init_prob = defaultdict(lambda: 0) # {init tag: #}
    emit_prob = defaultdict(lambda: defaultdict(lambda: 0)) # {tag: {word: # }}
    trans_prob = defaultdict(lambda: defaultdict(lambda: 0)) # {tag0:{tag1: # }}

    # number of unique words/tags seen in training data for tag T
    VT_emit = {} # {tag1 : number of unique words seen for tag1, ...}
    # total number of words in training data for tag T
    nT_emit = {} # {tag1 : total number of words in tag1 training data, ...}

    # dict of happax words and their corresponding tags (or an intermediate step to getting there)
    hapax = {}
    words = []
    
    total_tags = 0
    # TODO: (I)
    # Input the training set, output the formatted probabilities according to data statistics.
    for sentence in sentences:
        previous_tag = None
        for word,tag in sentence:
            # init prob section
            init_prob[tag] += 1
            total_tags += 1
            # emit prob section
            if tag in emit_prob:
                if word in emit_prob[tag]:
                    emit_prob[tag][word] += 1 
                else:
                    emit_prob[tag][word] = 1
                    VT_emit[tag] += 1
            else:
                emit_prob[tag][word] = 1
                VT_emit[tag] = 1
            if tag in nT_emit:
                nT_emit[tag] += 1
            else:
                nT_emit[tag] = 1
            # trans prob section
            if previous_tag != None:
                if previous_tag in trans_prob:
                    if tag in trans_prob[previous_tag]:
                        trans_prob[previous_tag][tag] += 1
                    else: 
                        trans_prob[previous_tag][tag] = 1
                else:
                    trans_prob[previous_tag][tag] = 1
            previous_tag = tag
            hapax[word] = tag
            words.append(word) 

    # Get hapax probs
    hapax_tag_probs = hapax_handler(words, hapax)

    # make init probabilities
    for tag in init_prob:
        init_prob[tag] /= total_tags

    # Probability smoothing ((count + alpha)/(n + alpha(V+1)))
    for tag in emit_prob:
        tag_dict = emit_prob[tag]
        hapax_multiplier = 0
        if tag in hapax_tag_probs:
            hapax_multiplier = hapax_tag_probs[tag]
        else:
            hapax_multiplier = hapax_tag_probs["unseen_tag"]
        for word in tag_dict:
            tag_dict[word] = log((tag_dict[word] + hapax_multiplier*emit_epsilon)/(nT_emit[tag] + hapax_multiplier*emit_epsilon*(VT_emit[tag] + 1)))
        tag_dict["unseen_words"] = log((hapax_multiplier*emit_epsilon)/(nT_emit[tag] + hapax_multiplier*emit_epsilon*(VT_emit[tag] + 1)))

    for tag in emit_prob: #we want to initialize for all the tags
        nt_total = sum(trans_prob[tag].values())
        for next_tag in emit_prob:
            if next_tag in trans_prob[tag]:
                trans_prob[tag][next_tag] = log(trans_prob[tag][next_tag]/nt_total)
            else:
                trans_prob[tag][next_tag] = log(epsilon_for_pt)

    return init_prob, emit_prob, trans_prob

def viterbi_stepforward(i, word, prev_prob, prev_predict_tag_seq, emit_prob, trans_prob):
    """
    Does one step of the viterbi function
    :param i: The i'th column of the lattice/MDP (0-indexing)
    :param word: The i'th observed word
    :param prev_prob: A dictionary of tags to probs representing the max probability of getting to each tag at in the
    previous column of the lattice
    :param prev_predict_tag_seq: A dictionary representing the predicted tag sequences leading up to the previous column
    of the lattice for each tag in the previous column
    :param emit_prob: Emission probabilities
    :param trans_prob: Transition probabilities
    :return: Current best log probs leading to the i'th column for each tag, and the respective predicted tag sequences
    """
    log_prob = {} # This should store the log_prob for all the tags at current column (i)
    predict_tag_seq = {} # This should store the tag sequence to reach each tag at column (i)

    # TODO: (II)
    # implement one step of trellis computation at column (i)
    # You should pay attention to the i=0 special case.
    if i == 0:
        for tag in emit_prob:
            if word in emit_prob[tag]:
                log_prob[tag] = prev_prob[tag] + emit_prob[tag][word]
            else:
                log_prob[tag] = prev_prob[tag] + emit_prob[tag]["unseen_words"]
            predict_tag_seq[tag] = []
            
    else:
        for tag in emit_prob:
            max_prob = float('-inf')
            max_prob_tag = None
            for prev_tag in prev_prob:
                if word in emit_prob[tag]:
                    cur_prob = prev_prob[prev_tag] + emit_prob[tag][word] + trans_prob[prev_tag][tag]
                else:
                    cur_prob = prev_prob[prev_tag] + emit_prob[tag]["unseen_words"] + trans_prob[prev_tag][tag]
                if cur_prob > max_prob:
                    max_prob = cur_prob
                    max_prob_tag = prev_tag
            log_prob[tag] = max_prob
            predict_tag_seq[tag] = prev_predict_tag_seq[tag]
            predict_tag_seq[tag].append(max_prob_tag)

    return log_prob, predict_tag_seq

def viterbi_2(train, test):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    init_prob, emit_prob, trans_prob = training(train)
    
    predicts = []
    
    for sen in range(len(test)):
        sentence=test[sen]
        length = len(sentence)
        log_prob = {}
        predict_tag_seq = {}
        # init log prob
        for t in emit_prob:
            if t in init_prob:
                log_prob[t] = log(init_prob[t])
            else:
                log_prob[t] = log(epsilon_for_pt)
            predict_tag_seq[t] = []

        # forward steps to calculate log probs for sentence
        for i in range(length):
            log_prob, predict_tag_seq = viterbi_stepforward(i, sentence[i], log_prob, predict_tag_seq, emit_prob,trans_prob)
            
        tagged_sentence = []
        best_choice = float('-inf')
        best_choice_tag = None
        last_tag = None
        for tag in log_prob:
            if log_prob[tag] > best_choice:
                best_choice = log_prob[tag]
                best_choice_tag = tag

        index = len(predict_tag_seq[best_choice_tag])-1
        last_tag = best_choice_tag
        # index = len(sentence)-1
        while index >= 0:
            tagged_sentence.append((sentence[index], predict_tag_seq[best_choice_tag][index]))
            best_choice_tag = predict_tag_seq[best_choice_tag][index]
            index -= 1
        tagged_sentence.reverse()
        tagged_sentence.append((sentence[len(sentence)-1], last_tag))
        predicts.append(tagged_sentence)
        # predicts.append(predict_tag_seq)

    return predicts