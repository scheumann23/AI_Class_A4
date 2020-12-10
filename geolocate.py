from sklearn.feature_extraction.text import CountVectorizer
import numpy as np 
import re
import collections
import pandas as pd
import math

def train_bayes(train_file):
    file = open(train_file, 'r')

    tweets = file.readlines()
    tweets = [re.sub('\n', '', tweet) for tweet in tweets]

    targets = [tweet.split()[0] for tweet in tweets]
    text = [' '.join(tweet.split()[1:]) for tweet in tweets]

    # learn the counts of locations
    p_L = collections.Counter(targets)

    # learn counts for each word by location
    cv = CountVectorizer()
    X = cv.fit_transform(text)
    word_df = pd.DataFrame(X.toarray(), columns = cv.get_feature_names())
    word_df['loc_target_Neelan'] = targets
    word_df = word_df.groupby(by = 'loc_target_Neelan').sum()
    word_df['loc_counts_Neelan'] = [p_L[key] for key in sorted(p_L.keys())]

    return word_df


def read_test_file(test_file):
    file = open(test_file, 'r')

    tweets = file.readlines()
    tweets = [re.sub('\n', '', tweet) for tweet in tweets]

    targets = [tweet.split()[0] for tweet in tweets]
    text = [' '.join(tweet.split()[1:]) for tweet in tweets]

    return (targets, text)


def test_one_target(test_tweet, target, train_df):
    test_tweet = test_tweet.lower()
    tokenized_tweet = re.split(r'\W', test_tweet)
    score = math.log(train_df.loc[target]['loc_counts_Neelan'] / train_df['loc_counts_Neelan'].sum())
    for token in tokenized_tweet:
        if token in train_df.columns:
            score += math.log((train_df.loc[target][token] + 1) / (train_df.loc[target].sum() + 1))
        else:
            score += math.log(1 / (train_df.loc[target].sum() + 1))
    return (target, score)

def bayes_test(test_tweet, targets, train_df):
    best_score = -1000000000000000
    best_target = ''
    for loc in targets:
        pos_target, pos_score = test_one_target(test_tweet, loc, train_df)
        if pos_score > best_score:
            best_score = pos_score
            best_target = pos_target
    return best_target



