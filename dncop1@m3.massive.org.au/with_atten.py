
# coding: utf-8

# In[155]:


import numpy as np
import os
from keras.models import Model
from keras.layers import Dense, Embedding, Activation, Permute
from keras.layers import Input, Flatten, Dropout
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed, Bidirectional


# In[156]:


from custom_recurrents import AttentionDecoder
from tdd import _time_distributed_dense


# In[157]:


from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer


# In[158]:


from cc_train_val_test import get_upsampled_raw_dataset


# In[159]:


import pandas as pd


# In[160]:


upsampled_raw_dataset = get_upsampled_raw_dataset()


# In[161]:


t,v,te = upsampled_raw_dataset


# In[162]:


t=t.dropna()
v=v.dropna()
te=te.dropna()


# In[205]:


tclue = t.clue.apply(lambda x:list(x))
vclue = v.clue.apply(lambda x:list(x))
teclue = te.clue.apply(lambda x:list(x))


# In[206]:


tokenizer = Tokenizer(filters='')


# In[207]:


tokenizer.fit_on_texts(pd.concat([tclue,vclue,teclue]))


# In[208]:


tx = pad_sequences(tokenizer.texts_to_sequences(tclue),maxlen=50)
vx = pad_sequences(tokenizer.texts_to_sequences(vclue),maxlen=50)
tex = pad_sequences(tokenizer.texts_to_sequences(teclue),maxlen=50)


# In[209]:


pad_length = 50
n_chars = len(tokenizer.index_word)+1
n_labels = 12
encoder_units=256,
decoder_units=256,
trainable=True
return_probabilities=True


# In[210]:


tx.shape


# In[223]:


def simpleNMT(pad_length = 50,n_chars = len(tokenizer.index_word)+1,n_labels = 12,encoder_units=256,decoder_units=256,trainable=True,return_probabilities=True):
    """
    Builds a Neural Machine Translator that has alignment attention
    :param pad_length: the size of the input sequence
    :param n_chars: the number of characters in the vocabulary
    :param n_labels: the number of possible labelings for each character
    :param embedding_learnable: decides if the one hot embedding should be refinable.
    :return: keras.models.Model that can be compiled and fit'ed

    *** REFERENCES ***
    Lee, Jason, Kyunghyun Cho, and Thomas Hofmann. 
    "Neural Machine Translation By Jointly Learning To Align and Translate" 
    """
    input_ = Input(shape=(pad_length,), dtype='float32')
    input_embed = Embedding(n_chars, n_chars,
                            input_length=pad_length,
                            trainable=embedding_learnable,
                            weights=[np.eye(n_chars)],
                            name='OneHot')(input_)

    rnn_encoded = Bidirectional(LSTM(encoder_units, return_sequences=True),
                                name='bidirectional_1',
                                merge_mode='concat',
                                trainable=trainable)(input_embed)

    atten_decoder = AttentionDecoder(decoder_units,
                             name='attention_decoder_1',
                             output_dim=n_labels,
                             return_probabilities=return_probabilities,
                             trainable=trainable)(rnn_encoded)
    y_hat = Flatten()(atten_decoder)
    y_hat = Dense(12,activation='softmax')(y_hat)

    model = Model(inputs=input_, outputs=y_hat)

    return model


# In[224]:


model = simpleNMT()


# In[225]:


model.summary()


# In[232]:


model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['categorical_accuracy'])


# In[240]:


from keras.callbacks import ModelCheckpoint
import os
cp = ModelCheckpoint("./weights/NMT.{epoch:02d}-{val_loss:.2f}.hdf5",
                     monitor='val_loss',
                     verbose=0,
                     save_best_only=True,
                     save_weights_only=True,
                     mode='auto')

# create a directory if it doesn't already exist
if not os.path.exists('./weights'):
    os.makedirs('./weights/')


# In[242]:


model.fit(tx,ty,validation_data=(vx,vy),batch_size=32,epochs=20,callbacks=[cp])

