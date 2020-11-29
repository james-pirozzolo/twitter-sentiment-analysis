import tensorflow as tf
import numpy as np
from functools import reduce

import csv
import re
import time


def clean_tweet(tweet):
    """
    Take a tweet (string) as input and return a parsed, stemmed version of it
    :param tweet: a string representing a tweet
    :return: an array of each cleaned word in the string
    """
    # remove punctuation
    no_punc_tweet = re.sub(r'[^\w\s]', '', tweet).lower()
    # split on whitespace, make each word lowercase
    return no_punc_tweet.split()


def tokenize_tweet(tweet_arr, vocab_dict):
    """
    simple helper method, converts a list of words to their respective ids
    :param tweet: an array of each cleaned word in a tweet
    :param vocab_dict: a word -> id dictionary
    :return: the tokenized array
    """
    return [vocab_dict[word] for word in tweet_arr]


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
    id = 0

    print('beginning preprocess for training data...')
    # a list of (tweets, sentiment)
    train_lst = []
    with open(train_file, 'r', encoding='ISO-8859-1') as csvfile:
        train_reader = csv.reader(csvfile)
        for row in train_reader:
            # tweet (in list form, sentiment)
            tweet_arr = clean_tweet(row[5])
            # add to vocab dict
            for word in set(tweet_arr):
                if word not in vocab_dict.keys():
                    vocab_dict[word] = id
                    id += 1
            train_lst.append((clean_tweet(row[5]), int(row[0])))
    print('finished preprocess for training data')

    print('beginning preprocess for testing data...')
    # a list of (tweets, sentiment)
    test_lst = []
    with open(test_file, 'r', encoding='ISO-8859-1') as csvfile:
        test_reader = csv.reader(csvfile)
        for row in test_reader:
            # tweet (in list form, sentiment)
            tweet_arr = clean_tweet(row[5])
            # add to voacb dict
            for word in set(tweet_arr):
                if word not in vocab_dict.keys():
                    vocab_dict[word] = id
                    id += 1
            test_lst.append((clean_tweet(row[5]), int(row[0])))
    print('finished preprocess for testing data')

    training_tokens = [(tokenize_tweet(tweet, vocab_dict), sentiment)
                       for tweet, sentiment in train_lst]
    testing_tokens = [(tokenize_tweet(tweet, vocab_dict), sentiment)
                      for tweet, sentiment in test_lst]
    return training_tokens, testing_tokens, vocab_dict


def main():
    # takes a little under 1 minute to run
    start = time.time()
    get_data("../data/train.csv", "../data/test.csv")
    end = time.time()
    print(f'took {end - start} seconds to run')


if __name__ == '__main__':
    main()
