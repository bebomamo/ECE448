"""
Part 4: Here should be your best version of viterbi, 
with enhancements such as dealing with suffixes/prefixes separately
"""

import math
from collections import defaultdict, Counter
from math import log

epsilon_for_pt = 1e-5
emit_epsilon = 1e-5 

def type_identifier(word):
    if (is_number(word[0]) or is_number(word[-1])):
        return 'NUMERICAL'
    elif len(word) < 4:
        return 'VS'
    elif len(word) < 10:
        if word[-1] == 's':
            return 'SHORT_S'
        else:
            return 'SHORT'
    else:
        if word[-1] == 's':
            return 'LONG_S'
        else:
            return 'LONG'

def is_number(character):
    if character == 0 or character == 1 or character == 2 or character == 3 or character == 4 or character == 5 or character == 6 or character == 7 or character == 8 or character == 9:
        return True
    else:
        return False

def hapax_handler(words, hapax, emit_prob):
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
    hapax_tag_probs = defaultdict(lambda: defaultdict(lambda: 0))
    total_hapax_tags = 0
    for word in hapax_words: #recover the tags and gather types for the hapax words
        hapax_type = type_identifier(word)
        sorted_hapax[word] = (hapax[word], hapax_type)

    for word,(tag, htype) in sorted_hapax.items(): #count tag-type frequencies amongst hapax
        total_hapax_tags += 1
        print((tag, htype))
        if tag in hapax_tag_probs:
                if htype in hapax_tag_probs[tag]:
                        hapax_tag_probs[tag][htype] += 1
                else:
                        hapax_tag_probs[tag][htype] = 1
        else:
            hapax_tag_probs[tag][htype] = 1

    for tag in emit_prob:
        if tag not in hapax_tag_probs:
            hapax_tag_probs[tag]['NUMERICAL'] = 1
            hapax_tag_probs[tag]['VS'] = 1
            hapax_tag_probs[tag]['SHORT_S'] = 1
            hapax_tag_probs[tag]['SHORT'] = 1
            hapax_tag_probs[tag]['LONG_S'] = 1
            hapax_tag_probs[tag]['LONG'] = 1
        else:
            if 'NUMERICAL' not in hapax_tag_probs[tag]:
                hapax_tag_probs[tag]['NUMERICAL'] = 1
            if 'VS' not in hapax_tag_probs[tag]:
                hapax_tag_probs[tag]['VS'] = 1
            if 'SHORT_S' not in hapax_tag_probs[tag]:
                hapax_tag_probs[tag]['SHORT_S'] = 1
            if 'SHORT' not in hapax_tag_probs[tag]:
                hapax_tag_probs[tag]['SHORT'] = 1
            if 'LONG_S' not in hapax_tag_probs[tag]:
                hapax_tag_probs[tag]['LONG_S'] = 1
            if 'LONG' not in hapax_tag_probs[tag]:
                hapax_tag_probs[tag]['LONG'] = 1

    for tag in hapax_tag_probs: #divide by the total_hapax_tags
        for htype in hapax_tag_probs[tag]:
                hapax_tag_probs[tag][htype] = hapax_tag_probs[tag][htype]/total_hapax_tags
    hapax_tag_probs["unseen_tag"]["UNSEEN_TYPE"] = 1 / total_hapax_tags


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
    hapax_tag_probs = hapax_handler(words, hapax, emit_prob)

    # make init probabilities
    for tag in init_prob:
        init_prob[tag] /= total_tags

    # Probability smoothing ((count + alpha)/(n + alpha(V+1)))
    for tag in emit_prob:
        tag_dict = emit_prob[tag]
        hapax_multiplier = hapax_tag_probs[tag]
        tag_type_total = sum(hapax_multiplier.values())
        for word in tag_dict:
            tag_dict[word] = log((tag_dict[word] + tag_type_total*emit_epsilon)/(nT_emit[tag] + tag_type_total*emit_epsilon*(VT_emit[tag] + 1)))        
        tag_dict['NUMERICAL'] = log((hapax_multiplier['NUMERICAL']*emit_epsilon)/(nT_emit[tag] + hapax_multiplier['NUMERICAL']*emit_epsilon*(VT_emit[tag] + 1)))        
        tag_dict['VS'] = log((hapax_multiplier['VS']*emit_epsilon)/(nT_emit[tag] + hapax_multiplier['VS']*emit_epsilon*(VT_emit[tag] + 1)))        
        tag_dict['SHORT_S'] = log((hapax_multiplier['SHORT_S']*emit_epsilon)/(nT_emit[tag] + hapax_multiplier['SHORT_S']*emit_epsilon*(VT_emit[tag] + 1)))        
        tag_dict['SHORT'] = log((hapax_multiplier['SHORT']*emit_epsilon)/(nT_emit[tag] + hapax_multiplier['SHORT']*emit_epsilon*(VT_emit[tag] + 1)))        
        tag_dict['LONG_S'] = log((hapax_multiplier['LONG_S']*emit_epsilon)/(nT_emit[tag] + hapax_multiplier['LONG_S']*emit_epsilon*(VT_emit[tag] + 1)))
        tag_dict['LONG'] = log((hapax_multiplier['LONG']*emit_epsilon)/(nT_emit[tag] + hapax_multiplier['LONG']*emit_epsilon*(VT_emit[tag] + 1)))
        

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
                htype = type_identifier(word)
                log_prob[tag] = prev_prob[tag] + emit_prob[tag][htype]
            predict_tag_seq[tag] = []
            
    else:
        for tag in emit_prob:
            max_prob = float('-inf')
            max_prob_tag = None
            for prev_tag in prev_prob:
                if word in emit_prob[tag]:
                    cur_prob = prev_prob[prev_tag] + emit_prob[tag][word] + trans_prob[prev_tag][tag]
                else:
                    htype = type_identifier(word)
                    cur_prob = prev_prob[prev_tag] + emit_prob[tag][htype] + trans_prob[prev_tag][tag]
                if cur_prob > max_prob:
                    max_prob = cur_prob
                    max_prob_tag = prev_tag
            log_prob[tag] = max_prob
            predict_tag_seq[tag] = prev_predict_tag_seq[tag]
            predict_tag_seq[tag].append(max_prob_tag)

    return log_prob, predict_tag_seq

def viterbi_3(train, test):
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