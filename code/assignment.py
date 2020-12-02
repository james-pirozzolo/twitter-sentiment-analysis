import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from preprocess import get_data, pad_corpus, convert_to_id, clean_tweet

class Model(tf.keras.Model):
    def __init__(self, vocab_size):
        """
        The Model class predicts the sentiment of a tweet 
        :param vocab_size: The number of unique words in the data
        """

        super(Model, self).__init__()

        # TODO: initialize vocab_size, emnbedding_size
        self.vocab_size = vocab_size
        self.embedding_size = 300
        #self.learning_rate = 0.01
        self.batch_size = 250 
        # number of output classes 
        self.num_classes = 2
        # LSTM units 
        self.units = 150
        initial_learning_rate = 0.01
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=100000,
            decay_rate=0.96,
            staircase=True)
            
        # define network parameters and optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        # TODO: initialize embeddings and forward pass weights (weights, biases)
        #self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        self.embedding_matrix = tf.keras.layers.Embedding(self.vocab_size, self.embedding_size, mask_zero=True)
        self.lstm = tf.keras.layers.LSTM(self.units, return_sequences=True, return_state=True)
        self.dense_1 = tf.keras.layers.Dense(self.num_classes, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(self.num_classes, activation='softmax')

    
    def call(self, inputs, initial_state):
        """
        ...

        :param inputs: batch of tweets of shape (batch_size, tweet_size)
        :param initial_state: 2-d array of shape (batch_size, rnn_size) as a tensor
        :return: the batch element probabilities as a tensor, a final_state (Note 1: If you use an LSTM, the final_state will be the last two RNN outputs, 
        Note 2: We only need to use the initial state during generation)
        using LSTM and only the probabilites as a tensor and a final_state as a tensor when using GRU 
        """
        # get the embeddings of the inputs 
        embedding = self.embedding_matrix(inputs) # shape (batch_size, tweet_size, embedding_size)
        # apply the LSTM layer forward pass
        lstm_out, state_1, state_2 = self.lstm(embedding) # shape lstm_out (batch_size, tweet_size, units)
        # because we want to average all vectors to determine sentiment, we reduce_mean on the lstm output
        outputs = tf.reduce_mean(lstm_out, axis=1) # shape outputs (batch_size, units)
        dense_1_out = self.dense_1(outputs)
        probabilities = self.dense_2(dense_1_out)
        return probabilities, (state_1, state_2)
    
    def loss(self, probs, labels):
        """
        Calculates average cross entropy sequence to sequence loss of the prediction

        :param logits: a matrix of shape (batch_size, tweet_size, vocab_size) as a tensor
        :param labels: matrix of shape (batch_size, tweet_size) containing the labels
        :return: the loss of the model as a tensor of size 1
        """
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, probs)
        return tf.reduce_mean(loss)

    def accuracy(self, probs, labels, print_outputs=False):
        """
        Calculates the accuracy in testing a sequence of tweets through our model

        :param probs: probabilities matrix of shape (batch_size, tweet_size) ?
        :param labels: labels matrix of shape (batch_size, tweet_size)
        :return: accuracy of the prediction on the batch of tweets
        """
        decoded_symbols = tf.argmax(input=probs, axis=1)
        if print_outputs:
            print("probabilities: \n")
            print(probs)
            print("LABELS")
            print(labels)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(decoded_symbols, labels), dtype=tf.float32))
        return accuracy

def train(model, train_inputs, train_labels):
    """
    Runs through one epoch - all training examples.

    :param model: the initilized model to use for forward and backward pass
    :param train_inputs: train inputs (all inputs for training) of shape (num_inputs, tweet_size)
    :param train_labels: train labels (all labels for training) of shape (num_labels, tweet_size)
    :return: None
    """
    # # shuffle our train data
    # rows, columns = tf.shape(train_inputs)
    # indices = tf.random.shuffle(np.arange(rows))
    # train_inputs = tf.gather(train_inputs, indices) 
    # train_labels = tf.gather(train_labels, indices)
    num_epochs = 10
    for epoch in range(num_epochs):
        print("Training Epoch: " + str(epoch))
        num_batch = 1
        for i in range(0, len(train_inputs), model.batch_size):
            print("Training batch: " + str(num_batch))
            # batching inputs and labels 
            ibatch = train_inputs[i: i+model.batch_size] # ibatch shape: (batch_size, max_length=50)
            lbatch = train_labels[i: i+model.batch_size]  # shape (batch_size)   
            with tf.GradientTape() as tape:
                # forward pass, returning probabilities
                probs, _ = model.call(ibatch, initial_state=None)
                # computing loss
                loss = model.loss(probs, lbatch)
            # updating gradients
            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            num_batch += 1
    return

def test(model, test_inputs, test_labels):
    """
    Runs through one epoch - all testing examples

    :param model: the trained model to use for prediction
    :param test_inputs: train inputs (all inputs for testing) of shape (num_inputs,)
    :param test_labels: train labels (all labels for testing) of shape (num_labels,)
    :returns: perplexity of the test set
    """
    total_loss = 0
    total_acc = 0
    num_batches = 0
    i = 0
    while (i+model.batch_size) < len(test_inputs):
        print("Testing batch: " + str(num_batches + 1))
        # batching inputs and labels
        ibatch = test_inputs[i: i+model.batch_size]
        lbatch = test_labels[i: i+model.batch_size]
        # forward pass, returning probabilities
        probs, _ = model.call(ibatch, initial_state=None)
        # computing loss 
        loss = model.loss(probs, lbatch)
        total_loss += loss
        # computing accuracy
        if i == 0:
            acc = model.accuracy(probs, lbatch, True)
        else:
            acc = model.accuracy(probs, lbatch)
        total_acc += acc
        #incrementing counters
        num_batches += 1
        i += model.batch_size
    # computing average accuracy and loss 
    avg_acc = total_acc/num_batches
    avg_loss = total_loss/num_batches
    return avg_acc, avg_loss

def repl(model, vocab):
    print("welcome to the interactive repl!")
    print("Please input tweets to find out their sentiment!")
    print("type :exit to quit out")
    while True:
        raw_tweet = input ("> ")
        if raw_tweet == ':exit':
            break
        # going to need to preprocess 'tweet'
        cleaned_tweet = clean_tweet(raw_tweet)
        padded_tweet = pad_corpus([cleaned_tweet])
        tweet = convert_to_id(padded_tweet, vocab)
        probs, _ = model.call(tweet, initial_state=None)

        #print out bar
        #arr = probs.numpy()
        #bar_length = 80
        #positive = int(round(arr[1] * bar_length))
        #print('[' + '='*positive + '-'*(bar_length - positive)+']')

        print(probs)

def main():
    # TO-DO: Pre-process and vectorize the data
    #train_inputs, train_labels, test_inputs, test_labels, vocab_dict = get_data(
     #   '../data/train.csv', '../data/test.csv')
    train_inputs, train_labels, test_inputs, test_labels, vocab_dict = get_data(
        '../data/train_mini.csv', '../data/test.csv')

    # TODO: initialize model and tensorflow variables
    model = Model(len(vocab_dict))

    # TODO: Set-up the training step
    train(model, train_inputs, train_labels)

    # TODO: Set up the testing steps
    accuracy, _ = test(model, test_inputs, test_labels)

    # print the accuracy!! 
    print(f'Accuracy is {accuracy.numpy()}!')

    repl(model,vocab_dict)

if __name__ == '__main__':
    main()
