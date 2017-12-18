#coding=utf-8
''' Import theano and numpy '''
import theano
import numpy as np
execfile('00_readingInput.py')
# python3
# exec(open('00_readingInput.py').read())

''' Import keras to build a DL model '''
from keras.models import Sequential
from keras.layers.core import Dense, Activation

# print 'Building a model whose loss function is categorical_crossentropy'
''' For categorical_crossentropy '''
model = Sequential()
model.add(Dense(128,input_dim=200))
model.add(Activation('sigmoid'))
model.add(Dense(256))
model.add(Activation('sigmoid'))
model.add(Dense(5))
model.add(Activation('softmax'))

''' Set up the optimizer '''
from keras.optimizers import SGD, Adam, RMSprop, Adagrad
sgd = SGD(lr=0.01,momentum=0.0,decay=0.0,nesterov=False)

''' Compile model with specified loss and optimizer '''
# (Do!) 指定 loss function
model.compile(	loss='categorical_crossentropy',
				optimizer=sgd,
				metrics=['accuracy'])

''' set the size of mini-batch and number of epochs'''
batch_size = 16
nb_epoch = 30

'''Fit models and use validation_split=0.1 '''
# (Do!) 指定 batch_size, nb_epoch, shuffle or not 
# 		與 validation_split 的比例
history = model.fit( X_train, #X_train, Y_train,
					 Y_train, # output
					 batch_size=batch_size, # batch_size
					 nb_epoch=nb_epoch, # nb_epoch
					 shuffle=True, # shuffle
					 validation_split=0.1, # validation_split
					 verbose=1)	

'''Access the loss and accuracy in every epoch'''
loss	= history.history.get('loss')
acc 	= history.history.get('acc')

''' Visualize the loss and accuracy of both models'''
import matplotlib.pyplot as plt
plt.figure(0)
plt.subplot(121)
plt.plot(range(len(loss)), loss,label='CE')
plt.title('Loss')
plt.legend(loc='upper left')
plt.subplot(122)
plt.plot(range(len(acc)), acc,label='CE')
plt.title('Accuracy')
plt.savefig('00_firstModel.png',dpi=300,format='png')
plt.close()
# print 'Result saved into 00_lossFuncSelection.png'
