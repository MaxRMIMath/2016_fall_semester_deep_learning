#coding=utf-8
''' Import theano and numpy '''
import theano
import numpy as np
execfile('00_readingInput.py')
# python3
# exec(open('00_readingInput.py').read())

''' EarlyStopping '''
# (Do!) 從 keras.callbacks 中 import EerlyStopping 
# 		並設定 monitor 與 patience
from keras.callbacks import EarlyStopping
early_stopping=EarlyStopping(monitor='val_loss',patience=3)


''' set the size of mini-batch and number of epochs'''
batch_size = 16
nb_epoch = 50

''' Import keras to build a DL model '''
from keras.models import Sequential
from keras.layers.core import Dense, Activation

print 'Building a model whose optimizer=adam, activation function=softplus'
model_adam = Sequential()
model_adam.add(Dense(128, input_dim=200))
model_adam.add(Activation('softplus'))
model_adam.add(Dense(256))
model_adam.add(Activation('softplus'))
model_adam.add(Dense(5))
model_adam.add(Activation('softmax'))

''' Setting optimizer as Adam '''
from keras.optimizers import Adam
model_adam.compile(loss= 'categorical_crossentropy',
              		optimizer='Adam',
              		metrics=['accuracy'])

'''Fit models and use validation_split=0.1 '''
# (Do!) 在 fit 時加入 callbacks
#  		把先前設定的 EarlyStopping 加入
history_adam = model_adam.fit(X_train, Y_train,
							batch_size=batch_size,
							nb_epoch=nb_epoch,
							verbose=0,
							shuffle=True,
                    		validation_split=0.1,
                            callbacks=[early_stopping]
                    		# 加入 callbacks
                    		)

loss_adam = history_adam.history.get('loss')
acc_adam = history_adam.history.get('acc')
val_loss_adam = history_adam.history.get('val_loss')
val_acc_adam = history_adam.history.get('val_acc')

''' Visualize the loss and accuracy of both models'''
import matplotlib.pyplot as plt
plt.figure(0)
plt.subplot(121)
plt.plot(range(len(loss_adam)), loss_adam,label='Training')
plt.plot(range(len(val_loss_adam)), val_loss_adam,label='Validation')
plt.title('Loss')
plt.legend(loc='upper left')
plt.subplot(122)
plt.plot(range(len(acc_adam)), acc_adam,label='Training')
plt.plot(range(len(val_acc_adam)), val_acc_adam,label='Validation')
plt.title('Accuracy')
plt.savefig('07_earlystopping.png',dpi=300,format='png')
plt.close()
# print 'Result saved into 07_earlystopping.png'
