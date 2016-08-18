import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, Flatten
from keras.preprocessing.sequence import pad_sequences
import numpy as np

from timeit import default_timer as timer

tokens = {'A':0, 'C':1, 'G':2, 'T':3}
batch_size = 64
n_epochs = 5

def read_data(filename):

    with open(filename, 'r') as h:
        data = pd.read_excel(h)

    # filter out NaNs
    df = data.dropna(0)

    # shuffle the data
    df = df[['Status', 'DNASeq']]
    df.reindex(np.random.permutation(df.index))

    a = df.as_matrix()
    status, dna = a[:,0], a[:,1]

    # encode the labels: 0 - accepted, 1 - cancelled
    y = np.array([item == 'cancelled' for item in status],
        dtype=np.int8)

    # encode the dna strings
    x = []
    for s in dna:
        s_encoded = np.array([tokens[t] for t in s], dtype=np.int8)
        x.append(s_encoded)

    return x, y

def create_model(embed_dim=4):
    ''' Simplistic LSTM model for sequence classification.

    The architecture:
        - embedding layer
        - LSTM unidirectional layer
        - 2 Dense layers
    
    We are taking only last hidden activation.  It is not practical 
    for the long sequences. 

    Further steps:
        - average across hidden activations over time
        - add bidirectional LSTM layers
        - stack LSTM layers
        - try to use convolution instead of Embedding 

    '''

    # expected input data shape: (batch_size, timesteps)
    # expected output data shape: (batch_size,)
    
    model = Sequential()
    model.add(Embedding(embed_dim+1,8))
    model.add(LSTM(16, return_sequences=False,
               input_shape=(None, embed_dim+1)))

    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    return model

def split_train_test(x, y, train_size=0.7):
    part = int(len(x)*train_size)

    return x[:part], y[:part], x[part:], y[part:]

def train(model, x, y):
    x_train, y_train, x_test, y_test = split_train_test(x, y, 0.7)

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    n_batches = len(x_train)//batch_size

    for e in range(n_epochs):

        print "Epoch:", e

        for i in range(n_batches):
            t0 = timer()

            bx = x_train[i*batch_size:(i+1)*batch_size]
            by = y_train[i*batch_size:(i+1)*batch_size]

            bx = pad_sequences(bx, dtype=np.int8)
            stats = model.train_on_batch(bx, by)

            print 'Iteration: %d, loss: %.4f, accuracy: %.4f, time: %.2f' % (i, stats[0], stats[1], timer() - t0) 

    # Testing

    x_test_pad = pad_sequences(x_test, dtype=np.int8)
    stats = model.evaluate(x_test_pad, y_test)

    print 'Test loss:', stats[0]
    print 'Test accuracy:', stats[1]

    y_pred = model.predict_proba(x_test_pad).flatten()
    # todo: estimate ROC/AUC score
        
if __name__ == '__main__':
    x, y = read_data('data.xlsx')
    model = create_model()
    train(model, x, y)

