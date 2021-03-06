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
    return np.array( text, dtype=np.uint8 )

def main():
    corpus = shakespeare.load_data()
    # need a 256-character set with 1 extra 
    corpus = corpus[:len(corpus)-(len(corpus)%256)+1]
    xs = corpus[:-1]
    ys = corpus[1:]
    assert not len(xs)%256, len(xs)%256
    assert not len(ys)%256

    divisor = len(corpus)//256 //5 * 3 * 256
    train_x, train_y = xs[:divisor].reshape((-1, 256)), ys[:divisor].reshape((-1, 256))
    test_x, test_y = xs[divisor:].reshape((-1, 256)), ys[divisor:].reshape((-1, 256))


    inputs = 256
    
    model = Sequential()
    
    # Embedding doesn't seem to do what I want, I want something that 
    # is going to expand an index (8-bit character value) into a 
    # 256-bit vector, but that *doesn't* appear to be the effect here..
    model.add(Embedding(256, 256)) 
    model.add(LSTM(256, 256))
    # Pooling layer needed?
    model.add(Dropout(.8))
    model.add(Dense(256, inputs))
#    model.add(LSTM(128, 128))
#    model.add(LSTM(128, inputs))
#    model.add(Activation('softmax'))
    model.compile(loss='MSE', optimizer='SGD', class_mode="categorical")
    print('Compiled')


    model.fit(
        train_x, # inputs
        train_y, # training outputs
        batch_size=8, 
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
