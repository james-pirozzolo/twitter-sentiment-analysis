import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from preprocess import get_data

import pickle

class Model(tf.keras.Model):
    def __init__(self, vocab_size):
        """
        The Model class computes the sentiment predictions for a batch of tweets 

        :param vocab_size: The number of unique words in the data
        """

        super(Model, self).__init__()

        # Initializing hyperparameters
        self.vocab_size = vocab_size
        self.embedding_size = 300
        self.learning_rate = 0.0005
        self.batch_size = 256
        # number of output classes
        self.num_classes = 2
        # LSTM units
        self.units = 150

        # Initializing Adam optimizer
        self.optimizer = tf.keras.optimizers.Adam(
           learning_rate=self.learning_rate)

        # Initializing Keras layers: embedding, LSTM, and dense layers
        self.embedding_matrix = tf.keras.layers.Embedding(self.vocab_size, self.embedding_size, mask_zero=True)
        self.lstm = tf.keras.layers.LSTM(self.units, return_sequences=True, return_state=True)
        self.dense_1 = tf.keras.layers.Dense(self.num_classes, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(self.num_classes, activation='softmax')

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
    
    def loss(self, probs, labels):
        """
        Calculates the average loss of sentiment predictions for tweets in a given forward pass

        :param probs: a matrix of shape (batch_size, num_classes) as a tensor
        :param labels: matrix of shape (batch_size,) containing the labels
        :return: the loss of the model as a tensor of size 1
        """
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, probs)
        return tf.reduce_mean(loss)

    def accuracy(self, probs, labels):
        """
        Calculates the batch accuracy of sentiment predictions

        :param probs: probabilities matrix of shape (batch_size, num_classes) 
        :param labels: labels matrix of shape (batch_size,)
        :return: accuracy of the prediction on the batch of tweets
        """
        decoded_symbols = tf.argmax(input=probs, axis=1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(decoded_symbols, labels), dtype=tf.float32))
        return accuracy

def train(model, train_inputs, train_labels):
    """
    Runs through all training examples and trains the model batch by batch

    :param model: the initilized model to use for forward and backward pass
    :param train_inputs: train inputs (all inputs for training) of shape (num_inputs, max_length)
    :param train_labels: train labels (all labels for training) of shape (num_labels,)
    :return: None
    """
    batch_sz = model.batch_size
    for i in range(0, len(train_inputs), batch_sz):
        print("Training batch: " + str(i//batch_sz))
        # batching inputs and labels 
        ibatch = train_inputs[i: i+batch_sz] 
        lbatch = train_labels[i: i+batch_sz]   
        with tf.GradientTape() as tape:
            # forward pass, returning probabilities
            probs, _ = model.call(ibatch, initial_state=None)
            # computing loss
            loss = model.loss(probs, lbatch)
        # updating gradients
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return None   

def test(model, test_inputs, test_labels):
    """
    Runs through all testing examples and tests the model batch by batch

    :param model: the trained model to use for prediction
    :param test_inputs: test inputs (all inputs for testing) of shape (num_inputs, max_length)
    :param test_labels: test labels (all labels for testing) of shape (num_labels,)
    :returns: accuracy and average loss of the model on the test set 
    """
    total_loss = 0
    total_acc = 0
    num_batches = 0
    batch_sz = model.batch_size
    for i in range(0, len(test_inputs), batch_sz):
        print("Testing batch: " + str(num_batches + 1))
        # batching inputs and labels
        ibatch = test_inputs[i: i+batch_sz]
        lbatch = test_labels[i: i+batch_sz]
        # forward pass, returning probabilities
        probs, _ = model.call(ibatch, initial_state=None)
        # computing loss 
        loss = model.loss(probs, lbatch)
        total_loss += loss
        # computing accuracy
        acc = model.accuracy(probs, lbatch)
        total_acc += acc
        #incrementing counters
        num_batches += 1
    # computing average accuracy and loss 
    avg_acc = total_acc/num_batches
    avg_loss = total_loss/num_batches
    return avg_acc, avg_loss

def save_model(model, vocab):
    np.save('../saved_model/embedding.npy', model.embedding_matrix)
    np.save('../saved_model/lstm.npy', model.lstm)
    np.save('../saved_model/dense_1.npy', model.dense_1)
    np.save('../saved_model/dense_2.npy', model.dense_2)
    np.save('../saved_model/vocab.npy', vocab)

def main():
    # Pre-process the data
    train_inputs, train_labels, test_inputs, test_labels, vocab_dict = get_data(
        '../data/train_200k.csv', '../data/test.csv')
    # Initialize the model and tensorflow variables 
    model = Model(len(vocab_dict))

    train(model, train_inputs, train_labels)
    accuracy, _ = test(model, test_inputs, test_labels)
    print(f'Accuracy is {accuracy.numpy()}!')

    save_model(model, vocab_dict)

if __name__ == '__main__':
    main()
