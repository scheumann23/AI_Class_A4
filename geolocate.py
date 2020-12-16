import numpy as np 
import re
from collections import defaultdict, Counter
import math
import copy
import sys
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from statistics import mode

# train the Naive Bayes model
def train_bayes(train_file, model_file):
    # open the training file and parse it into targets (i.e. locations) and the rest of the text
    file = open(train_file, 'r')

    tweets = file.readlines()
    tweets = [re.sub('\n', '', tweet) for tweet in tweets]

    targets = [tweet.split()[0] for tweet in tweets]
    text = [tweet.lower().split()[1:] for tweet in tweets]

    # create a dictionary for to track all of the words in a given location
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

    # create another dictionary to track each location the the words show up in 
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

    # create a dictionary that counts how many times each location appears 
    p_L = Counter(targets)

    # convert the loc_word_dict from counts to percentages
    for value in loc_word_dict.values():
        total = sum(value.values())
        for key in value.keys():
            value[key] = value[key] / total

    # narrow down the word_loc_dict dictionary to only words that appear more than 5 times
    word_loc_dict2 = {}
    for key in word_loc_dict.keys():
        if sum(word_loc_dict[key].values()) >= 5:
            word_loc_dict2[key] = word_loc_dict[key]

    # convert from counts to percentages
    for value in word_loc_dict2.values():
        total = sum(value.values())
        for key in value.keys():
            value[key] = [value[key] / total, value[key]]
    
    # convert from counts to percentages
    total = sum(p_L.values())
    for key in p_L.keys():
        p_L[key] = p_L[key] / total

    # find the top 5 words for each location based on highest P(L|word) and then by highest count
    top_words = {}
    for loc in set(targets):
        words = []
        for word in word_loc_dict2.keys():
            if loc in word_loc_dict2[word].keys():
                words.append([word, word_loc_dict2[word][loc]])
        words = sorted(words, key = lambda x: (x[1][0], x[1][1]), reverse = True)[0:5]
        words = [word[0] for word in words]
        top_words[loc] = words

    # print out the top 5 words to the screen 
    for key in top_words.keys():
            ws = ', '.join(top_words[key])
            out = f'The top 5 words for {key} are: {ws}'
            print(out)
    
    # save the model into a pickle file for testing later
    full_model = {'model_type': 'bayes', 'loc_word_dict': loc_word_dict, 'p_L': p_L}

    f = open(model_file, 'wb')
    pickle.dump(full_model, f)
    f.close

# read the testing file and tokenize the tweets
def read_test_file_bayes(test_file):
    file = open(test_file, 'r')

    tweets = file.readlines()
    tweets = [re.sub('\n', '', tweet) for tweet in tweets]

    targets = [tweet.split()[0] for tweet in tweets]
    text = [' '.join(tweet.split()[1:]) for tweet in tweets]

    return (targets, text)

# calculate the probability of one location 
def test_one_target(test_tweet, target, loc_word_dict, p_L):
    tokenized_tweet = test_tweet.lower().split()
    score = math.log(p_L[target])
    for token in tokenized_tweet:
        if token in loc_word_dict[target].keys():
            score += math.log(loc_word_dict[target][token])
        else:
            score += math.log(1 / 100000)
    return (target, score)

# cycle through all the possible locations and choose the one with the highest probability
def bayes_test(test_tweet, targets, loc_word_dict, p_L):
    best_score = -1000000000000000
    best_target = ''
    for loc in targets:
        pos_target, pos_score = test_one_target(test_tweet, loc, loc_word_dict, p_L)
        if pos_score > best_score:
            best_score = pos_score
            best_target = pos_target
    return best_target

# cycle through all of the test tweets and make predictions
def predict_bayes(test_text, test_targets, loc_word_dict, p_L, output_file):
    correct = 0
    total = 0
    predictions = []
    for i in range(len(test_text)):
        prediction = bayes_test(test_text[i], set(test_targets), loc_word_dict, p_L)
        predictions.append((prediction, test_targets[i], test_text[i]))
        total += 1
        if prediction == test_targets[i]:
            correct += 1

    # save the results to the output file
    f = open(output_file, "w")
    for line in predictions:
        for word in line:
            f.write(word)
            f.write(' ')
        f.write('\n')
    f.close
        
    score = correct / total
    print(score)

# read in the training file for the dtree algorithm
def read_train_file_dtree(train_file):
    file = open(train_file, 'r')

    tweets = file.readlines()
    tweets = [re.sub('\n', '', tweet) for tweet in tweets]
    
    # tokenize the tweets 
    targets = [tweet.split()[0] for tweet in tweets]
    text = [' '.join(tweet.split()[1:]) for tweet in tweets]
    text2 = [re.findall(r'\w\w+', tweet.lower())[1:] for tweet in tweets]

    # similar to the Naive Bayes work create a method to pick the best words for each location
    word_loc_dict = {}
    for i in range(len(targets)):
        for word in text2[i]:
            if word in word_loc_dict.keys():
                if targets[i] in word_loc_dict[word].keys():
                    word_loc_dict[word][targets[i]] += 1
                else:
                    word_loc_dict[word][targets[i]] = 1
            else:
                word_loc_dict[word] = {targets[i]: 1}

    word_loc_dict2 = {}
    for key in word_loc_dict.keys():
        if sum(word_loc_dict[key].values()) >= 20:
            word_loc_dict2[key] = word_loc_dict[key]

    for value in word_loc_dict2.values():
        total = sum(value.values())
        for key in value.keys():
            value[key] = [value[key] / total, value[key]]

    top_words = []
    for loc in set(targets):
        words = []
        for word in word_loc_dict2.keys():
            if loc in word_loc_dict2[word].keys():
                words.append([word, word_loc_dict2[word][loc]])
        words = sorted(words, key = lambda x: (x[1][0], x[1][1]), reverse = True)[0:50]
        words = [word[0] for word in words]
        top_words += words

    # create a matrix representation of the tweets
    cv = CountVectorizer(binary=True)
    X = cv.fit_transform(text)

    # only choose the top words determined above
    top_words = sorted([cv.get_feature_names().index(word) for word in set(top_words) if word in cv.get_feature_names()])

    X = X.toarray()[:, top_words]
    word_list = [cv.get_feature_names()[index] for index in top_words]


    return (X, np.array(targets), word_list)    

# based on: https://stackoverflow.com/questions/3229419/how-to-pretty-print-nested-dictionaries
def print_dict(d, indent_count=0):
    if isinstance(d, dict):
        print('\t' * indent_count + '+ ' + d['node'])
        if indent_count < 3:
            print_dict(d['left'], indent_count+1)
            print_dict(d['right'], indent_count+1)
    else:
        print('\t' * indent_count + '+ ' + d)
    indent_count = 3 

# train the decision tree
def train_dtree(matrix, labels, word_list, dtree, min_leaves, max_depth, depth = 1):
    # if only one label remains return that label
    if len(set(labels)) == 1:
        return labels[0]
    # if we have reached the minimum number of labels in the leaf return the mode of the lables
    elif len(labels) <= min_leaves:
        return mode(labels)
    # if we have reached the max depth of the tree return the mode of the remaining labels
    elif depth == max_depth:
        return mode(labels)
    else:
        # determine which label is the best to split one
        split_word, split_word_index = best_split(matrix, labels, word_list)
        dtree['node'] = split_word
        attr = matrix[:,split_word_index]

        # left will mean the tweet has the word, right will mean the tweet does not
        
        left_filter = [attr == 1]
        right_filter = [attr == 0]

        # split the remaining matrix as well as the labels into two halves
        left_matrix = np.delete(matrix[tuple(left_filter)], split_word_index, axis = 1)
        left_labels = labels[tuple(left_filter)]

        right_matrix = np.delete(matrix[tuple(right_filter)], split_word_index, axis = 1)
        right_labels = labels[tuple(right_filter)]

        # remove the split word so it can't be used again
        word_list.pop(split_word_index)

        empty_dict_left = {'node': '', 'left': {}, 'right': {}}
        empty_dict_right = {'node': '', 'left': {}, 'right': {}}

        # rerun the whole thing for the left and the right side of the tree
        dtree['left'] = train_dtree(left_matrix, left_labels, word_list, empty_dict_left, min_leaves, max_depth, depth+1)
        dtree['right'] = train_dtree(right_matrix, right_labels, word_list, empty_dict_right, min_leaves, max_depth, depth+1)

    return dtree

# calculate the entropy after splitting on a given word
def entropy(attr, labels):
    ones = labels[attr == 1]
    zeroes = labels[attr == 0]
    one_length = len(ones)
    zero_length = len(zeroes)

    ones_entropy = 0
    for label in set(labels):
        label_count = sum([1 for word in ones if word == label])
        if one_length == 0:
            prop = 0
        else:
            prop = label_count / one_length
        ones_entropy += 0 if prop == 0 else (-prop * math.log(prop, 2))

    zeroes_entropy = 0
    for label in set(labels):
        label_count = sum([1 for word in zeroes if word == label])
        if zero_length == 0:
            prop = 0
        else:
            prop = label_count / zero_length
        zeroes_entropy += 0 if prop == 0 else (-prop * math.log(prop, 2))

    total_entropy = ((ones_entropy * one_length) + (zeroes_entropy * zero_length)) / (one_length + zero_length)
    
    return total_entropy

# determine the best word to split on based on the word that has the lowest entropy
def best_split(matrix, labels, word_list):
    best_choice = ''
    best_entropy = 1000000000000
    for i in range(len(word_list)):
        attr = matrix[:,i]
        ent = entropy(attr, labels)
        if ent < best_entropy:
            best_entropy = ent
            best_choice = word_list[i]
            best_choice_index = i
    return (best_choice, best_choice_index)

# read in the testing file and tokenize the tweets
def read_test_file_dtree(test_file):
    file = open(test_file, 'r')

    tweets = file.readlines()
    tweets = [re.sub('\n', '', tweet) for tweet in tweets]

    targets = [tweet.split()[0] for tweet in tweets]
    text = [re.findall(r'\w\w+', tweet.lower())[1:] for tweet in tweets]

    return (targets, text)

# test a single tweet to see how the model would classify it
def dtree_test(test_tweet, dtree):
    # if we are at a string that means we have reached a leaf so return the value
    if isinstance(dtree, str):
        return dtree
    # otherwise keep searching through the tree
    else:
        if dtree['node'] in test_tweet:
            output = dtree_test(test_tweet, dtree['left'])
        else:
            output = dtree_test(test_tweet, dtree['right'])
    return output

# predict the location for all of the tweet, save the output and print the accuracy
def predict_dtree(test_text, test_targets, dtree, output_file):
    correct = 0
    total = 0
    predictions = []
    for i in range(len(test_text)):
        prediction = dtree_test(test_text[i], dtree)
        predictions.append((prediction, test_targets[i], ' '.join(test_text[i])))
        total += 1
        if prediction == test_targets[i]:
            correct += 1

    f = open(output_file, "w")
    for line in predictions:
        for word in line:
            f.write(word)
            f.write(' ')
        f.write('\n')
    f.close
        
    score = correct / total
    print(score)



if __name__ == "__main__":
    train_or_test = sys.argv[1]

    if train_or_test == 'train':
        bayes_or_dtree = sys.argv[2]
        train_file = sys.argv[3]
        model_file = sys.argv[4]
        print(f'Training a {bayes_or_dtree} model...')
        if bayes_or_dtree == 'bayes':
            train_bayes(train_file, model_file)
        elif bayes_or_dtree == 'dtree':
            matrix, labels, word_list = read_train_file_dtree(train_file)
            dtree = train_dtree(matrix, labels, word_list, {'node': '', 'left': {}, 'right': {}}, 10, 10)
            print_dict(dtree, 0)
            full_model = {'model_type': 'dtree', 'dtree_dict': dtree}
            with open(model_file, 'wb') as f:
                pickle.dump(full_model, f)
                f.close
        else:
            print('Error: Please select either bayes or dtree')

    elif train_or_test == 'test':
        model_file = sys.argv[2]
        test_input_file = sys.argv[3]
        test_output_file = sys.argv[4]

        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        bayes_or_dtree = model['model_type']
        
        print(f'Making predictions using the {bayes_or_dtree} model...')
        if bayes_or_dtree == 'bayes':
            loc_word_dict = model['loc_word_dict']
            p_L = model['p_L']
            test_targets, test_text = read_test_file_bayes(test_input_file)
            predict_bayes(test_text, test_targets, loc_word_dict, p_L, test_output_file)
        elif bayes_or_dtree == 'dtree':
            dtree = model['dtree_dict']
            test_targets, test_text = read_test_file_dtree(test_input_file)
            predict_dtree(test_text, test_targets, dtree, test_output_file)
