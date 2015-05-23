"""Generate Shakespeare-like texts using an LSTM trained on Shakespeare

See: http://karpathy.github.io/2015/05/21/rnn-effectiveness/

For the original article
"""
from __future__ import absolute_import
from __future__ import print_function
import numpy as np

from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.datasets import shakespeare
from theano import config
from keras.preprocessing.text import Tokenizer

def charmap_array( text,  char_map ):
    y = len(char_map)
    data = np.zeros((len(text), y), dtype=config.floatX)
    for i, char in enumerate(text):
        index = char_map[char]
        data[i][index] = 1.0
    return data

def main():
    corpus, chars = shakespeare.load_data()
    corpus = corpus[:1000]
    inputs = len(chars)
    char_map = dict([(c, i) for i, c in enumerate(sorted(chars))])
    
    model = Sequential()
    
    model.add(LSTM(inputs, inputs))
    model.add(Dropout(.8))
    model.add(Dense(256, inputs))
#    model.add(LSTM(128, 128))
#    model.add(LSTM(128, inputs))
    model.add(Activation('sigmoid'))
    model.compile(loss='MSE', optimizer='SGD', class_mode="categorical")
    print('Compiled')

    divisor = len(corpus)//4*3
    train_x, train_y = charmap_array(corpus[:divisor], char_map), charmap_array(corpus[1:divisor+1], char_map)
    test_x, test_y = charmap_array(corpus[divisor:-1], char_map), charmap_array(corpus[divisor+1:], char_map)
    


    model.fit(
        train_x, # inputs
        train_y, # training outputs
        batch_size=256, 
        nb_epoch=400, 
        validation_split=0.1, show_accuracy=True
    )
    model.save_weights( 'shakespeare_weights.hd5' )
    score = model.evaluate(
        test_x, 
        test_y, 
        batch_size=200
    )
    print('Test score:', score)

#    classes = model.predict_classes(X_test, batch_size=batch_size)
#    acc = np_utils.accuracy(classes, y_test)
#    print('Test accuracy:', acc)
#
if __name__ == "__main__":
    main()
