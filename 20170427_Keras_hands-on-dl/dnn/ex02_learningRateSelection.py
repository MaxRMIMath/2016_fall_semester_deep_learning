#coding=utf-8
''' Import theano and numpy '''
import theano
import numpy as np
execfile('00_readingInput.py')
# python3
# exec(open('00_readingInput.py').read())

''' set the size of mini-batch and number of epochs'''
batch_size = 16
nb_epoch = 30

''' Import keras to build a DL model '''
from keras.models import Sequential
from keras.layers.core import Dense, Activation

# print 'Building model using SGD(lr=0.1)'
''' 1. Model using large learning rate '''
model_large = Sequential()
model_large.add(Dense(128, input_dim=200))
model_large.add(Activation('sigmoid'))
model_large.add(Dense(256))
model_large.add(Activation('sigmoid'))
model_large.add(Dense(5))
model_large.add(Activation('softmax'))

''' set the learning rate of SGD optimizer to 0.1 '''
from keras.optimizers import SGD, Adam, RMSprop, Adagrad
# (Do!) 設定一個較大的 learning rate
sgd_large = SGD(lr=0.1 , # 填入一個數字, ex: 0.1
				momentum=0.0,decay=0.0,nesterov=False)

model_large.compile(loss= 'categorical_crossentropy',
              optimizer=sgd_large,
              metrics=['accuracy'])

history_large = model_large.fit(X_train, Y_train,
								batch_size=batch_size,
								nb_epoch=nb_epoch,
								verbose=0,
								shuffle=True,
                    			validation_split=0.1)

loss_large = history_large.history.get('loss')
acc_large = history_large.history.get('acc')

# print 'Building model using SGD(lr=0.01)'
''' 2. Model using median learning rate '''
model_median = Sequential()
model_median.add(Dense(128, input_dim=200))
model_median.add(Activation('sigmoid'))
model_median.add(Dense(256))
model_median.add(Activation('sigmoid'))
model_median.add(Dense(5))
model_median.add(Activation('softmax'))


''' set the learning rate of SGD optimizer to 0.01 '''
# (Do!) 設定一個比 sgd_large 還小 learning rate
sgd_median = SGD(lr=0.01, # 填入一個數字, ex: 0.1
				momentum=0.0,decay=0.0,nesterov=False)

model_median.compile(loss= 'categorical_crossentropy',
              optimizer=sgd_median ,
              metrics=['accuracy'])

history_median = model_median.fit(X_train, Y_train,
						batch_size=batch_size,
						nb_epoch=nb_epoch,
						verbose=0,
						shuffle=True,
                    	validation_split=0.1)

loss_median = history_median.history.get('loss')
acc_median = history_median.history.get('acc')

# print 'Building model using SGD(lr=0.001)'
''' 3. Model using small learning rate '''
model_small = Sequential()
model_small.add(Dense(128, input_dim=200))
model_small.add(Activation('sigmoid'))
model_small.add(Dense(256))
model_small.add(Activation('sigmoid'))
model_small.add(Dense(5))
model_small.add(Activation('softmax'))

''' set the learning rate of SGD optimizer to 0.001 '''
# (Do!) 設定一個比 sgd_median還小 learning rate
sgd_small = SGD(lr=0.001,momentum=0.0,decay=0.0,nesterov=False)

model_small.compile(loss= 'categorical_crossentropy',
              optimizer=sgd_small,
              metrics=['accuracy'])

history_small = model_small.fit(X_train, Y_train,
						batch_size=batch_size,
						nb_epoch=nb_epoch,
						verbose=0,
						shuffle=True,
                    	validation_split=0.1)

loss_small = history_small.history.get('loss')
acc_small = history_small.history.get('acc')

import matplotlib.pyplot as plt
plt.figure(0)
plt.subplot(121)
plt.plot(range(len(loss_large)), loss_large,label='lr=0.1')
plt.plot(range(len(loss_median)), loss_median,label='lr=0.01')
plt.plot(range(len(loss_small)), loss_small,label='lr=0.001')
plt.title('Loss')
plt.legend(loc='upper left')
plt.subplot(122)
plt.plot(range(len(acc_large)), acc_large,label='lr=0.1')
plt.plot(range(len(acc_median)), acc_median,label='lr=0.01')
plt.plot(range(len(acc_small)), acc_small,label='lr=0.001')
plt.title('Accuracy')
plt.savefig('02_learningRateSelection.png',dpi=300,format='png')
plt.close()
# print 'Result saved into 02_learningRateSelection.png'
