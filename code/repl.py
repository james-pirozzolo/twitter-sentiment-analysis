from preprocess import pad_corpus, convert_to_id, clean_tweet
import pickle
import numpy as np
from loaded_model import Model as LoadedModel
import sys

# def repl(model, vocab):
#     print("welcome to the interactive repl!")
#     print("Please input tweets to find out their sentiment!")
#     print("type :exit to quit out")
#     while True:
#         raw_tweet = input("> ")
#         if raw_tweet == ':exit':
#             break
#         probs = get_tweet_sentiment(model,vocab,raw_tweet)
#         print(probs)

def load_model(filepath):
    '''
    Loads and returns all required layers for calling model
    param filepath: file path to files storing layers
    return: layers required for loaded mode
    '''
    embedding = np.load(filepath + '/embedding.npy', allow_pickle='TRUE').item()
    lstm = np.load(filepath + '/lstm.npy', allow_pickle='TRUE').item()
    dense_1 = np.load(filepath + '/dense_1.npy', allow_pickle='TRUE').item()
    dense_2 = np.load(filepath + '/dense_2.npy', allow_pickle='TRUE').item()
    vocab = np.load(filepath + '/vocab.npy', allow_pickle='TRUE').item()
    return (embedding, lstm, dense_1, dense_2, vocab)

def get_tweet_sentiment(model,vocab,raw_tweet):
    # going to need to preprocess 'tweet'
    cleaned_tweet = clean_tweet(raw_tweet)
    padded_tweet = pad_corpus([cleaned_tweet])
    tweet = convert_to_id(padded_tweet, vocab)
    probs, _ = model.call(tweet, initial_state=None)
    # print(probs)
    return probs

def get_tweet_sentiment_from_filepath(raw_tweet, filepath):
    # returns negative sentiment
    embedding, lstm, dense_1, dense_2, vocab = load_model(filepath)
    model = LoadedModel(embedding, lstm, dense_1, dense_2)
    probs = get_tweet_sentiment(model, vocab, raw_tweet)
    return probs.numpy()[0][0]

def main():
    # print('here!')
    # filepath = '../saved_model'
    # tweet = "i hate lebron james"
    prbs = get_tweet_sentiment_from_filepath(sys.argv[1], sys.argv[2])
    # prbs = get_tweet_sentiment_from_filepath(tweet, filepath)
    print(prbs)
    # print('hello world')
    # sys.stdout.flush()

if __name__ == '__main__':
    main()
