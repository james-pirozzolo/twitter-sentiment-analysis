import tensorflow as tf
import numpy as np
from tensorflow.keras import Model

class Model(tf.keras.Model):
    def __init__(self, embedding, lstm, dense_1, dense_2):
        """
        The Model class computes the sentiment predictions for a batch of tweets 
        :param vocab_size: The number of unique words in the data
        """

        super(Model, self).__init__()

        # Initializing Keras layers: embedding, LSTM, and dense layers
        self.embedding_matrix = embedding
        self.lstm = lstm
        self.dense_1 = dense_1
        self.dense_2 = dense_2

    def call(self, inputs, initial_state):
        """
        Performs the forward pass on a batch of tweets to generate the sentiment probabilities.
        This returns a tensor of shape [batch_size, num_classes], where each row is a
        probability distribution over the sentiment for tweet.
        :param inputs: batch of tweets of shape (batch_size, max_length)
        :param initial_state: 2-d array of shape (batch_size, rnn_size) as a tensor  --> might want to remove this 
        :return: the batch probabilities tensor, and the last two LSTM states. 
        """
        # get the embeddings of the inputs 
        embedding = self.embedding_matrix(inputs) # shape (batch_size, tweet_size, embedding_size)
        # apply the LSTM layer forward pass
        lstm_out, state_1, state_2 = self.lstm(embedding) # shape lstm_out (batch_size, tweet_size, units)
        # because we want to average all vectors to determine sentiment, we reduce_mean on the lstm output
        outputs = tf.reduce_mean(lstm_out, axis=1) # shape outputs (batch_size, units)
        # apply the dense layers to get probabilities
        dense_1_out = self.dense_1(outputs)
        probabilities = self.dense_2(dense_1_out)
        return probabilities, (state_1, state_2)