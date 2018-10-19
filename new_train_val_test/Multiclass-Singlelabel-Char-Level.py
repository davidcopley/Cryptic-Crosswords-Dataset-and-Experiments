
# coding: utf-8

# In[7]:


import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)


# In[8]:


import pandas as pd
import math
import keras
from keras.layers import Dense,Embedding, Flatten, Conv1D, GlobalMaxPooling1D, LSTM, Bidirectional, Dropout
from keras.preprocessing.text import text_to_word_sequence,Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from new_train_val_test import cc_train_val_test


# In[9]:


train,val,test = cc_train_val_test.get_upsampled_raw_dataset()


# In[13]:


train.clue = train.clue.apply(lambda x:list(x))
val.clue = val.clue.apply(lambda x:list(x))
test.clue = test.clue.apply(lambda x:list(x))


# In[15]:


tokenizer = Tokenizer(filters="")


# In[16]:


tokenizer.fit_on_texts(pd.concat([train,val,test]).clue)


# In[21]:


train_x = train.clue
val_x = val.clue
test_x = test.clue


# In[36]:


train_x = pad_sequences(tokenizer.texts_to_sequences(train_x),50)
val_x = pad_sequences(tokenizer.texts_to_sequences(val_x),50)
test_x = pad_sequences(tokenizer.texts_to_sequences(test_x),50)


# In[37]:


train_y = train[train.columns[4:]]
val_y = val[train.columns[4:]]
test_y = test[test.columns[4:]]


# In[38]:


filepath="./models/1xBilstm-{epoch:02d}-{loss:.2f}-{categorical_accuracy:.2f}-{val_loss:.2f}-{val_categorical_accuracy:.2f}-singlelabel.hdf5"
saveModelCallBack = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
from keras.callbacks import Callback

class TestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))

callbacks_list = [saveModelCallBack,tbCallBack,TestCallback((test_x, test_y))]


# In[39]:


model = keras.Sequential()
model.add(Embedding(len(tokenizer.index_word)+1, 128))
model.add(Bidirectional(LSTM(128, dropout=0.5)))
model.add(Dense(12, activation='softmax'))


# In[40]:


model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['categorical_accuracy'])


# In[41]:


model.summary()


# In[ ]:


history = model.fit(train_x,train_y ,validation_data=(test_x,test_y), batch_size=32, epochs=15,shuffle=True,callbacks=callbacks_list,initial_epoch=1)

