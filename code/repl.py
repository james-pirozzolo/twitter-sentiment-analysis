from preprocess import pad_corpus, convert_to_id, clean_tweet
from keras.models import load_model
import pickle

def repl(model, vocab):
    print("welcome to the interactive repl!")
    print("Please input tweets to find out their sentiment!")
    print("type :exit to quit out")
    while True:
        raw_tweet = input("> ")
        if raw_tweet == ':exit':
            break
        # going to need to preprocess 'tweet'
        cleaned_tweet = clean_tweet(raw_tweet)
        padded_tweet = pad_corpus([cleaned_tweet])
        tweet = convert_to_id(padded_tweet, vocab)
        probs, _ = model.call(tweet, initial_state=None)

        print(probs)

def main():
    # access model
    model = load_model('../model/sentiment_model.h5')
    f = open('../model/vocab.pkl', 'rb')
    repl(model, pickle.load(f))
    f.close()

if __name__ == '__main__':
    main()
