import theano
import numpy as np

execfile('00_readingInput.py')

from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
from keras.regularizers import l2

model=Sequential()
model.add(Dense(60,input_dim=200,W_regularizer=l2(0.01)))
model.add(Activation('relu'))
model.add(Dropout(0.1))
#model.add(Dense(75,W_regularizer=l2(0.01)))
#model.add(Activation('softplus'))
#model.add(Dropout(0.4))
model.add(Dense(5,W_regularizer=l2(0.01)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

history=model.fit(X_train,Y_train,
                    batch_size=20,
                    nb_epoch=100,
                    verbose=0,
                    shuffle=True,
                    validation_split=0.1)

loss=history.history.get('loss')
acc=history.history.get('acc')
val_loss=history.history.get('val_loss')
val_acc=history.history.get('val_acc')

import matplotlib.pyplot as plt
plt.figure(0)
plt.subplot(121)
plt.plot(range(len(loss)),loss,label='loss')
plt.plot(range(len(val_loss)),val_loss,label='val_loss')
plt.title('loss')
plt.legend(loc='upper left')
plt.subplot(122)
plt.plot(range(len(acc)),acc,label='acc')
plt.plot(range(len(val_acc)),val_acc,label='val_acc')
plt.title('acc')
plt.legend(loc='upper left')
plt.savefig('my_pkgo.png',dpi=300,format='png')
plt.close()

