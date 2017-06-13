from __future__ import print_function
import numpy as np
np.random.seed(42)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Convolution1D, MaxPooling1D
from keras.datasets import imdb
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Embedding
max_features = 20000
maxlen = 100
embedding_size = 128

# Convolution
filter_length = 5
nb_filter = 64
pool_length = 4

# LSTM
lstm_output_size = 70

# Training
batch_size = 30
nb_epoch = 100
test_ratio = 0.6

'''
Note:
batch_size is highly sensitive.
Only 2 epochs are needed as the dataset is very small.
'''

print('Loading data...')
(X_train, y_train), (X_Test, y_Test) = imdb.load_data(num_words=max_features)
X_test = X_Test[:int(test_ratio*len(X_Test))]
X_validation = X_Test[int(test_ratio*len(X_Test)):]
y_test = y_Test[:int(test_ratio*len(y_Test))]
y_validation = y_Test[int(test_ratio*len(y_Test)):]

print(len(X_train), 'train sequences')
print(len(X_validation), 'validation sequences')
print(len(X_test), 'test sequences')

print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_validation = sequence.pad_sequences(X_validation, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_validation shape:', X_validation.shape)
print('X_test shape:', X_test.shape)

print('Build model...')

model = Sequential()
model.add(Embedding(max_features, embedding_size, input_length=maxlen))
model.add(Dropout(0.25))
model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))
model.add(MaxPooling1D(pool_length=pool_length))
model.add(LSTM(lstm_output_size))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adagrad',
              metrics=['accuracy'])
print(model.summary())
plot_model(model, to_file='model_cnn_lstm.png')

model_fname = 'model_cnn_lstm.hdf5'
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=10, verbose=0, mode='auto')
model_checkpoint = ModelCheckpoint(model_fname, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
callbacks=[early_stopping, model_checkpoint]

print('Train...')
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          validation_data=(X_validation, y_validation), callbacks=callbacks)
score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
