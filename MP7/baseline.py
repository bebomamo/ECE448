"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""

def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    # tag_freq_map = {word1 : {tag1 : tag1 frequency, tag2: tag2 frequency}, word2 : {tag1 : tag1 frequency, tag2 : tag2 frequency}, ...}
    tag_freq_map = dict()
    for sentence in train:
                for (word,tag) in sentence:
                        if word in tag_freq_map:
                                if tag in tag_freq_map[word]:
                                        tag_freq_map[word][tag]+=1
                                else:
                                        tag_freq_map[word][tag]=1
                        else:
                                tag_freq_map[word] = dict()
                                tag_freq_map[word][tag]=1
    # Now the map is filled but since we are using baseline, we only care about the most frequent tag for each word
    baseline_freq_map = dict()
    tag_map = dict()
    for key, tag_dict in tag_freq_map.items():
                max_tag_freq = 0
                max_tag = None
                for cur_tag,tag_freq in tag_dict.items():
                        if tag_freq > max_tag_freq: #find the max frequency tag for each word
                                max_tag_freq = tag_freq
                                max_tag = cur_tag
                        if cur_tag in tag_map: #tally the frequency of every tag across all words for unseen word tagging
                                tag_map[cur_tag] += tag_freq
                        else:
                                tag_map[cur_tag] = tag_freq
                baseline_freq_map[key] = max_tag
    max_tag_freq = 0 
    max_tag = None
    for tag,tag_freq in tag_map.items(): #computes the most seen tag for unseen words
                if tag_freq > max_tag_freq:
                        max_tag_freq = tag_freq
                        max_tag = tag
    test_output =  []
    for sentence in test:
                sentence_list = []
                for word in sentence:
                        if word in tag_freq_map:
                                sentence_list.append((word, baseline_freq_map[word]))
                        else:
                                sentence_list.append((word, max_tag))
                test_output.append(sentence_list)
    

    return test_output