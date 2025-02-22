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
from keras.layers.core import Dense, Dropout, Activation

print 'Building model using softplus as activation function'
''' Use other activation function '''
model_sp = Sequential()
model_sp.add(Dense(128, input_dim=200))
model_sp.add(Activation('softplus'))
# (Do!) 加入 activation function
model_sp.add(Dense(256))
model_sp.add(Activation('softplus'))
# (Do!) 加入 activation function
model_sp.add(Dense(5))
model_sp.add(Activation('softmax'))
# (Do!) 加入 activation function

''' Use SGD(lr=0.01) as the optimizer  '''
''' lr set to 0.01 according to 02_learningRateSelection.py '''
from keras.optimizers import SGD
sgd = SGD(lr=0.01,momentum=0.0,decay=0.0,nesterov=False)

model_sp.compile(loss= 'categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

history_sp = model_sp.fit(X_train, Y_train,
							batch_size=batch_size,
							nb_epoch=nb_epoch,
							verbose=0,
							shuffle=True,
                    		validation_split=0.1)

loss_sp = history_sp.history.get('loss')
acc_sp = history_sp.history.get('acc')

###### 對照組 #######
model_bm = Sequential()
model_bm.add(Dense(128, input_dim=200))
model_bm.add(Activation('relu'))
model_bm.add(Dense(256))
model_bm.add(Activation('relu'))
model_bm.add(Dense(5))
model_bm.add(Activation('softmax'))
from keras.optimizers import SGD, Adam, RMSprop, Adagrad
sgd = SGD(lr=0.01,momentum=0.0,decay=0.0,nesterov=False)
model_bm.compile(loss='categorical_crossentropy',
				optimizer=sgd,
				metrics=['accuracy'])
history_bm = model_bm.fit(X_train, Y_train,
							batch_size=batch_size,
							nb_epoch=nb_epoch,
							verbose=0,
							shuffle=True,
                    		validation_split=0.1)
loss_bm	= history_bm.history.get('loss')
acc_bm 	= history_bm.history.get('acc')
###### 對照組 #######

import matplotlib.pyplot as plt
plt.figure(0)
plt.subplot(121)
plt.plot(range(len(loss_sp)),loss_sp,label='Softplus')
plt.plot(range(len(loss_bm)),loss_bm,label='relu')
plt.title('Loss')
plt.legend(loc='upper left')
plt.subplot(122)
plt.plot(range(len(acc_sp)),acc_sp,label='Softplus')
plt.plot(range(len(acc_bm)),acc_bm,label='relu')
plt.title('Accuracy')
plt.savefig('03_activationFuncSelection.png',dpi=300,format='png')
plt.close()
