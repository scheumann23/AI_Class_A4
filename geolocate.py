import numpy as np 
import re
from collections import defaultdict, Counter
import math
import copy

def train_bayes(train_file):
    file = open(train_file, 'r')

    tweets = file.readlines()
    tweets = [re.sub('\n', '', tweet) for tweet in tweets]

    targets = [tweet.split()[0] for tweet in tweets]
    text = [tweet.lower().split()[1:] for tweet in tweets]

    loc_word_dict = {}
    for i in range(len(targets)):
        for word in text[i]:
            if targets[i] in loc_word_dict.keys():
                if word in loc_word_dict[targets[i]].keys():
                    loc_word_dict[targets[i]][word] += 1
                else:
                    loc_word_dict[targets[i]][word] = 1
            else:
                loc_word_dict[targets[i]] = {word: 1}

    word_loc_dict = {}
    for i in range(len(targets)):
        for word in text[i]:
            if word in word_loc_dict.keys():
                if targets[i] in word_loc_dict[word].keys():
                    word_loc_dict[word][targets[i]] += 1
                else:
                    word_loc_dict[word][targets[i]] = 1
            else:
                word_loc_dict[word] = {targets[i]: 1}

    p_L = Counter(targets)

    for value in loc_word_dict.values():
        total = sum(value.values())
        for key in value.keys():
            value[key] = value[key] / total

    word_loc_dict2 = {}
    for key in word_loc_dict.keys():
        if sum(word_loc_dict[key].values()) >= 5:
            word_loc_dict2[key] = word_loc_dict[key]

    for value in word_loc_dict2.values():
        total = sum(value.values())
        for key in value.keys():
            value[key] = [value[key] / total, value[key]]
    
    total = sum(p_L.values())
    for key in p_L.keys():
        p_L[key] = p_L[key] / total

    top_words = {}
    for loc in set(targets):
        words = []
        for word in word_loc_dict2.keys():
            if loc in word_loc_dict2[word].keys():
                words.append([word, word_loc_dict2[word][loc]])
        words = sorted(words, key = lambda x: (x[1][0], x[1][1]), reverse = True)[0:5]
        words = [word[0] for word in words]
        top_words[loc] = words

    for key in top_words.keys():
            ws = ', '.join(top_words[key])
            out = f'The top 5 words for {key} are: {ws}'
            print(out)
    

    return loc_word_dict, p_L

def read_test_file(test_file):
    file = open(test_file, 'r')

    tweets = file.readlines()
    tweets = [re.sub('\n', '', tweet) for tweet in tweets]

    targets = [tweet.split()[0] for tweet in tweets]
    text = [' '.join(tweet.split()[1:]) for tweet in tweets]

    return (targets, text)

def test_one_target(test_tweet, target, loc_word_dict, p_L):
    tokenized_tweet = test_tweet.lower().split()
    score = math.log(p_L[target])
    for token in tokenized_tweet:
        if token in loc_word_dict[target].keys():
            score += math.log(loc_word_dict[target][token])
        else:
            score += math.log(1 / 100000)
    return (target, score)

def bayes_test(test_tweet, targets, loc_word_dict, p_L):
    best_score = -1000000000000000
    best_target = ''
    for loc in targets:
        pos_target, pos_score = test_one_target(test_tweet, loc, loc_word_dict, p_L)
        if pos_score > best_score:
            best_score = pos_score
            best_target = pos_target
    return best_target


