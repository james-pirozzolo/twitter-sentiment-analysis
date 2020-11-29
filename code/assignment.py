import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from preprocess import get_data

class Model(tf.keras.Model):
    def __init__(self, vocab_size):
        """
        The Model class predicts the sentiment of a tweet 
        :param vocab_size: The number of unique words in the data
        """

        super(Model, self).__init__()

        # TODO: initialize vocab_size, emnbedding_size
        self.vocab_size = vocab_size
        # self.window_size = 20 
        self.embedding_size = 300 
        self.batch_size = 50 
        # number of output classes 
        self.num_classes = 2 
        # LSTM units 
        self.units = 150

        # TODO: initialize embeddings and forward pass weights (weights, biases)
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.001)

        self.embedding_matrix = tf.keras.layers.Embedding(vocab_size, self.embedding_size)
        self.lstm = tf.keras.layers.LSTM(self.units, return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(self.units) # check units of units (150)
        
        # might want to use Weights matrix and matmul 
        # self.W = tf.Variable(tf.random.truncated_normal([self.units, self.units], stddev=.1))
        # self.dense_2 = tf.keras.layers.Dense(vocab_size)
    
    def call(self, inputs, initial_state):
        """
        ...

        :param inputs: 
        :param initial_state: 2-d array of shape (batch_size, rnn_size) as a tensor
        :return: the batch element probabilities as a tensor, a final_state (Note 1: If you use an LSTM, the final_state will be the last two RNN outputs, 
        Note 2: We only need to use the initial state during generation)
        using LSTM and only the probabilites as a tensor and a final_state as a tensor when using GRU 
        """
        # get the embeddings of the inputs 
        embedding = self.embedding_matrix(inputs)
        # apply the LSTM layer forward pass
        lstm_out, state_1, state_2 = self.lstm(embedding)
        # because we want to average all vectors to determine sentiment, we reduce_mean on the lstm output
        outputs = tf.reduce_mean(lstm_out, axis=1)
        # apply the dense layer to get logits ((X*W)+b)
        logits = self.dense(outputs)
        # activation function to obtain probabilities 
        probabilites = tf.nn.softmax(logits)

        return probabilites, (state_1, state_2)
    
    def loss(self, probs, labels):
        """
        Calculates average cross entropy sequence to sequence loss of the prediction

        :param logits: a matrix of shape (batch_size, window_size, vocab_size) as a tensor
        :param labels: matrix of shape (batch_size, window_size) containing the labels
        :return: the loss of the model as a tensor of size 1
        """
        # loss = tf.keras.losses.sparse_categorical_crossentropy(
        #     labels, probs)
        # return tf.reduce_mean(loss)

def train(model, train_inputs, train_labels):
    """
    Runs through one epoch - all training examples.

    :param model: the initilized model to use for forward and backward pass
    :param train_inputs: train inputs (all inputs for training) of shape (num_inputs,)
    :param train_labels: train labels (all labels for training) of shape (num_labels,)
    :return: None
    """
    # # TODO: Fill in
    # window_inputs = []
    # window_labels = []
    # i=0
    # while (i+model.window_size-1) < len(train_inputs):
    #     iwindow = []
    #     lwindow= []
    #     for j in range(model.window_size):
    #         iwindow.append(train_inputs[i+j])
    #         lwindow.append(train_labels[i+j])
    #     window_inputs.append(iwindow)
    #     window_labels.append(lwindow)
    #     i+= model.window_size

    # x = 0
    # while (x + model.batch_size) < len(window_inputs):
    #     ibatch = window_inputs[x : x+model.batch_size]
    #     ibatch = np.asarray(ibatch)
    #     lbatch = window_labels[x : x+model.batch_size]
    #     with tf.GradientTape() as tape:
    #         probs, final_state = model.call(ibatch, initial_state=None)
    #         loss = model.loss(probs, lbatch)
    #     gradients = tape.gradient(loss, model.trainable_variables)
    #     model.optimizer.apply_gradients(
    #         zip(gradients, model.trainable_variables))
    #     x += model.batch_size
    # pass


def test(model, test_inputs, test_labels):
    """
    Runs through one epoch - all testing examples

    :param model: the trained model to use for prediction
    :param test_inputs: train inputs (all inputs for testing) of shape (num_inputs,)
    :param test_labels: train labels (all labels for testing) of shape (num_labels,)
    :returns: perplexity of the test set
    """
    # TODO: Fill in
    # NOTE: Ensure a correct perplexity formula (different from raw loss)
    # window_inputs = []
    # window_labels = []
    # i=0
    # while (i+model.window_size-1) < len(test_inputs):
    #     iwindow = []
    #     lwindow= []
    #     for j in range(model.window_size):
    #         iwindow.append(test_inputs[i+j])
    #         lwindow.append(test_labels[i+j])
    #     window_inputs.append(iwindow)
    #     window_labels.append(lwindow)
    #     i+= model.window_size
    
    # x = 0
    # total_loss = 0
    # num_batches = 0
    # while (x + model.batch_size) < len(window_inputs):
    #     ibatch = window_inputs[x : x+model.batch_size]
    #     ibatch = np.asarray(ibatch)
    #     lbatch = window_labels[x : x+model.batch_size]
    #     probs, final_state = model.call(ibatch, initial_state=None)
    #     loss = model.loss(probs, lbatch)

    #     total_loss += loss
    #     num_batches += 1
    #     x += model.batch_size
    # avg_loss = total_loss/num_batches
    # return np.exp(avg_loss)

def main():
    # TO-DO: Pre-process and vectorize the data
    train_data, test_data, vocab_dict = get_data(
        '../data/train.csv', '../data/test.csv')

    # TO-DO:  Separate your train and test data into inputs and labels
    train_inputs = train_data[0]
    train_labels = train_data[1]

    test_inputs = test_data[0]
    test_labels = test_data[1]
    # TODO: initialize model and tensorflow variables
    model = Model(len(vocab_dict))

    # TODO: Set-up the training step
    # train(model, train_inputs, train_labels)

    # # TODO: Set up the testing steps
    # perplexity = test(model, test_inputs, test_labels)

    # Print out perplexity
    # print(perplexity)

if __name__ == '__main__':
    main()