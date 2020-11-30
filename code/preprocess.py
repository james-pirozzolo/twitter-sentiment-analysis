import tensorflow as tf
import numpy as np
from functools import reduce

import csv
import re
import time


##########DO NOT CHANGE#####################
PAD_TOKEN = "*PAD*"
STOP_TOKEN = "*STOP*"
START_TOKEN = "*START*"
UNK_TOKEN = "*UNK*"
# max length in train is ~40
WINDOW_SIZE = 50
# WINDOW_SIZE = 140 (the max )
##########DO NOT CHANGE#####################

def read_data(file, vocab_dict, next_word_id, num_lines=None):
    """
    Reads in data from a file and returns a list of tweets and sentiments
    :param file: a string path to the file
    :param vocab_dict a dictionary that assigns ids to words
    :return: a tuple of a 2d array of # tweets x tweet array 
    and the sentiment for each tweet (in an array)
    """
    # a list of (tweets, sentiment)
    tweets = []
    sentiments = []
    with open(file, 'r', encoding='ISO-8859-1') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # tweet (in list form, sentiment)
            tweet_arr = clean_tweet(row[5])
            # add to vocab dict
            for word in set(tweet_arr):
                if word not in vocab_dict.keys():
                    vocab_dict[word] = next_word_id
                    next_word_id += 1
            tweets.append(clean_tweet(row[5]))
            sentiments.append(0) if int(row[0]) <= 2 else sentiments.append(1)
    return tweets, sentiments, next_word_id

def pad_corpus(raw_tweets):
    """
    arguments are lists of tweets. Returns padded tweets The
    text is given an initial "*STOP*".  All sentences are padded with "*STOP*" at
    the end.
    :param raw_tweets: list of tweets
    :return: list of padded tweets
    """
    TWEETS_padded = []
    for line in raw_tweets:
        padded_TWEET = line[:WINDOW_SIZE]
        padded_TWEET += [STOP_TOKEN] + [PAD_TOKEN] * (WINDOW_SIZE - len(padded_TWEET)-1)
        TWEETS_padded.append(padded_TWEET)
    return TWEETS_padded

def clean_tweet(tweet):
    """
    Take a tweet (string) as input and return a parsed, stemmed version of it
    :param tweet: a string representing a tweet
    :return: an array of each cleaned word in the string
    """
    # removing @mentions
    tweet = re.sub('@\S+', '', tweet)
    # remove hyperlinks
    tweet = re.sub('https?:\/\/\S+', '', tweet)
    # remove punctuation
    tweet = re.sub(r'[^\w\s]', '', tweet)
    # split on whitespace, make each word lowercase
    return tweet.lower().split()

def convert_to_id(tweets, vocab_dict):
    """
    simple helper method, converts a list of tweet arrays to their respective ids
    :param tweets: a list of padded tweet arrays of each cleaned word in a tweet
    :param vocab_dict: a word -> id dictionary
    :return: the corresponding np id array 
    """
    return np.stack([[vocab_dict[word] for word in tweet] for tweet in tweets])


def get_data(train_file, test_file):
    """
    Read and parse the train and test file line by line, then tokenize the sentences to build the train and test data separately.
    Create a vocabulary dictionary that maps all the unique tokens from your train and test data as keys to a unique integer value.
    Then vectorize your train and test data based on your vocabulary dictionary.
    : param train_file: Path to the training file.
    : param test_file: Path to the test file.
    : return: Tuple of train(1-d list or array with training words in vectorized/id form), test(1-d list or array with testing words in vectorized/id form), vocabulary(Dict containg index -> word mapping)
    """
    vocab_dict = {}
    vocab_dict[PAD_TOKEN] = 0
    vocab_dict[START_TOKEN] = 1
    vocab_dict[STOP_TOKEN] = 2
    vocab_dict[UNK_TOKEN] = 3
    word_id = 4
    print('download and clean training data...')
    train_raw_tweets, train_sentiments, word_id = read_data(train_file, vocab_dict, word_id)
    print('download and clean testing data...')
    test_raw_tweets, test_sentiments, word_id = read_data(train_file, vocab_dict, word_id)
    print('padding data...')
    train_tweet_pad = pad_corpus(train_raw_tweets)
    test_tweet_pad = pad_corpus(test_raw_tweets)
    print('vectorizing data...')
    train_tweets = convert_to_id(train_tweet_pad, vocab_dict)
    test_tweets = convert_to_id(test_tweet_pad, vocab_dict)

    return train_tweets, np.array(train_sentiments), test_tweets, np.array(test_sentiments), vocab_dict

def main():
    # takes a little under 1 minute to run
    start = time.time()
    # get_data("../data/test.csv", "../data/test.csv")
    get_data("../data/train_mini.csv", "../data/test.csv")
    # get_data("../data/train.csv", "../data/test.csv")
    end = time.time()
    print(f'took {end - start} seconds to run')


if __name__ == '__main__':
    main()
