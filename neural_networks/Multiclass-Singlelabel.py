
# coding: utf-8

# In[11]:


import pandas as pd
import math
import keras
from keras.layers import Dense,Embedding, Flatten, Conv1D, GlobalMaxPooling1D, LSTM, Bidirectional, Dropout
from keras.preprocessing.text import text_to_word_sequence,Tokenizer
from keras.preprocessing.sequence import pad_sequences


# In[12]:


df = pd.read_pickle("./cryptic_dataset/combined_fifteen_times_final_filtered.pickle")


# In[13]:


anagram_df = df[
    df.is_anagram &
    ~df.is_homophone &
    ~df.is_double &
    ~df.is_cryptic & 
    ~df.is_contain & 
    ~df.is_reverse & 
    ~df.is_alternate &
    ~df.is_init & 
    ~df.is_delete & 
    ~df.is_charade & 
    ~df['is_&lit'] & 
    ~df.is_hidden & 
    ~df.is_spoonerism & 
    ~df.is_palindrome
]


# In[14]:


anagram_df.sample(1000,random_state=1).to_csv('pure_anagrams.csv')


# In[15]:


homophone_df = df[
    ~df.is_anagram &
    df.is_homophone &
    ~df.is_double &
    ~df.is_cryptic & 
    ~df.is_contain & 
    ~df.is_reverse & 
    ~df.is_alternate &
    ~df.is_init & 
    ~df.is_delete & 
    ~df.is_charade & 
    ~df['is_&lit'] & 
    ~df.is_hidden & 
    ~df.is_spoonerism & 
    ~df.is_palindrome
]


# In[17]:


double_df = df[
    ~df.is_anagram &
    ~df.is_homophone &
    df.is_double &
    ~df.is_cryptic & 
    ~df.is_contain & 
    ~df.is_reverse & 
    ~df.is_alternate &
    ~df.is_init & 
    ~df.is_delete & 
    ~df.is_charade & 
    ~df['is_&lit'] & 
    ~df.is_hidden & 
    ~df.is_spoonerism & 
    ~df.is_palindrome
]


# In[18]:


cryptic_df = df[
    ~df.is_anagram &
    ~df.is_homophone &
    ~df.is_double &
    df.is_cryptic & 
    ~df.is_contain & 
    ~df.is_reverse & 
    ~df.is_alternate &
    ~df.is_init & 
    ~df.is_delete & 
    ~df.is_charade & 
    ~df['is_&lit'] & 
    ~df.is_hidden & 
    ~df.is_spoonerism & 
    ~df.is_palindrome
]


# In[19]:


contain_df = df[
    ~df.is_anagram &
    ~df.is_homophone &
    ~df.is_double &
    ~df.is_cryptic & 
    df.is_contain & 
    ~df.is_reverse & 
    ~df.is_alternate &
    ~df.is_init & 
    ~df.is_delete & 
    ~df.is_charade & 
    ~df['is_&lit'] & 
    ~df.is_hidden & 
    ~df.is_spoonerism & 
    ~df.is_palindrome
]


# In[20]:


reverse_df = df[
    ~df.is_anagram &
    ~df.is_homophone &
    ~df.is_double &
    ~df.is_cryptic & 
    ~df.is_contain & 
    df.is_reverse & 
    ~df.is_alternate &
    ~df.is_init & 
    ~df.is_delete & 
    ~df.is_charade & 
    ~df['is_&lit'] & 
    ~df.is_hidden & 
    ~df.is_spoonerism & 
    ~df.is_palindrome
]


# In[21]:


alternate_df = df[
    ~df.is_anagram &
    ~df.is_homophone &
    ~df.is_double &
    ~df.is_cryptic & 
    ~df.is_contain & 
    ~df.is_reverse & 
    df.is_alternate &
    ~df.is_init & 
    ~df.is_delete & 
    ~df.is_charade & 
    ~df['is_&lit'] & 
    ~df.is_hidden & 
    ~df.is_spoonerism & 
    ~df.is_palindrome
]


# In[22]:


init_df = df[
    ~df.is_anagram &
    ~df.is_homophone &
    ~df.is_double &
    ~df.is_cryptic & 
    ~df.is_contain & 
    ~df.is_reverse & 
    ~df.is_alternate &
    df.is_init & 
    ~df.is_delete & 
    ~df.is_charade & 
    ~df['is_&lit'] & 
    ~df.is_hidden & 
    ~df.is_spoonerism & 
    ~df.is_palindrome
]


# In[23]:


delete_df = df[
    ~df.is_anagram &
    ~df.is_homophone &
    ~df.is_double &
    ~df.is_cryptic & 
    ~df.is_contain & 
    ~df.is_reverse & 
    ~df.is_alternate &
    ~df.is_init & 
    df.is_delete & 
    ~df.is_charade & 
    ~df['is_&lit'] & 
    ~df.is_hidden & 
    ~df.is_spoonerism & 
    ~df.is_palindrome
]


# In[24]:


charade_df = df[
    ~df.is_anagram &
    ~df.is_homophone &
    ~df.is_double &
    ~df.is_cryptic & 
    ~df.is_contain & 
    ~df.is_reverse & 
    ~df.is_alternate &
    ~df.is_init & 
    ~df.is_delete & 
    df.is_charade & 
    ~df['is_&lit'] & 
    ~df.is_hidden & 
    ~df.is_spoonerism & 
    ~df.is_palindrome
]


# In[25]:


lit_df = df[
    ~df.is_anagram &
    ~df.is_homophone &
    ~df.is_double &
    ~df.is_cryptic & 
    ~df.is_contain & 
    ~df.is_reverse & 
    ~df.is_alternate &
    ~df.is_init & 
    ~df.is_delete & 
    ~df.is_charade & 
    df['is_&lit'] & 
    ~df.is_hidden & 
    ~df.is_spoonerism & 
    ~df.is_palindrome
]


# In[26]:


hidden_df = df[
    ~df.is_anagram &
    ~df.is_homophone &
    ~df.is_double &
    ~df.is_cryptic & 
    ~df.is_contain & 
    ~df.is_reverse & 
    ~df.is_alternate &
    ~df.is_init & 
    ~df.is_delete & 
    ~df.is_charade & 
    ~df['is_&lit'] & 
    df.is_hidden & 
    ~df.is_spoonerism & 
    ~df.is_palindrome
]


# In[27]:


spoonerism_df = df[
    ~df.is_anagram &
    ~df.is_homophone &
    ~df.is_double &
    ~df.is_cryptic & 
    ~df.is_contain & 
    ~df.is_reverse & 
    ~df.is_alternate &
    ~df.is_init & 
    ~df.is_delete & 
    ~df.is_charade & 
    ~df['is_&lit'] & 
    ~df.is_hidden & 
    df.is_spoonerism & 
    ~df.is_palindrome
]


# In[28]:


palindrome_df = df[
    ~df.is_anagram &
    ~df.is_homophone &
    ~df.is_double &
    ~df.is_cryptic & 
    ~df.is_contain & 
    ~df.is_reverse & 
    ~df.is_alternate &
    ~df.is_init & 
    ~df.is_delete & 
    ~df.is_charade & 
    ~df['is_&lit'] & 
    ~df.is_hidden & 
    ~df.is_spoonerism & 
    df.is_palindrome
]

# In[37]:


tokenizer = Tokenizer()
tokenizer.fit_on_texts(df.clue.tolist())


# In[29]:


cc_types_dfs = [anagram_df,homophone_df,double_df,cryptic_df,contain_df,reverse_df,alternate_df,init_df,delete_df,lit_df,hidden_df,spoonerism_df,palindrome_df]


# In[30]:


cc_types = 'is_anagram	is_homophone	is_double	is_cryptic	is_contain	is_reverse	is_alternate	is_init	is_delete	is_&lit	is_hidden	is_spoonerism	is_palindrome'.split('	')


# In[31]:


for df,cc_type in zip(cc_types_dfs,cc_types):
    df['category'] = cc_type


# In[32]:


def get_input_val_test(df):
    length = len(df)
    input_len = math.floor(length*0.7)
    val_len  = math.floor(length*0.2)
    test_len = math.floor(length*0.1)
    input_df = df[:input_len]
    val_df = df[input_len:input_len+val_len]
    test_df = df[input_len+val_len:]
    return input_df,val_df,test_df


# In[33]:


input_cc_types_df = pd.concat([get_input_val_test(df)[0] for df in cc_types_dfs]).sample(frac=1)
val_cc_types_df = pd.concat([get_input_val_test(df)[1] for df in cc_types_dfs]).sample(frac=1)
test_cc_types_df = pd.concat([get_input_val_test(df)[2] for df in cc_types_dfs]).sample(frac=1)


# In[34]:


max_size = input_cc_types_df.groupby('category').count().max()[0]


# In[35]:


lst = [input_cc_types_df]
for class_index, group in input_cc_types_df.groupby('category'):
    sample = group.sample(max_size-len(group), replace=True, )
    lst.append(sample)
upsampled_input_cc_types_df = pd.concat(lst)


# In[36]:


cc_input_df = upsampled_input_cc_types_df.drop('category',axis=1)
cc_val_df = val_cc_types_df.drop('category',axis=1).drop_duplicates()
cc_test_df = test_cc_types_df.drop('category',axis=1).drop_duplicates()





# In[38]:


cc_input_data = pad_sequences(tokenizer.texts_to_sequences(cc_input_df.clue.tolist()),maxlen=15)
cc_val_data = pad_sequences(tokenizer.texts_to_sequences(cc_val_df.clue.tolist()),maxlen=15)
cc_test_data = pad_sequences(tokenizer.texts_to_sequences(cc_test_df.clue.tolist()),maxlen=15)


# In[39]:


cc_input_data_out = cc_input_df[cc_input_df.columns[2:]] * 1
cc_val_data_out = cc_val_df[cc_val_df.columns[2:]] * 1
cc_test_data_out = cc_test_df[cc_test_df.columns[2:]] * 1


model = keras.Sequential()
model.add(Embedding(len(tokenizer.word_index)+1, 256))
model.add(Bidirectional(LSTM(256, dropout=0.5, recurrent_dropout=0.5)))
model.add(Dense(len(cc_types_dfs), activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['categorical_accuracy'])


# In[46]:


filepath="1xBilstm-{epoch:02d}-{val_loss:.2f}-{val_categorical_accuracy:.2f}-singlelabel.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
callbacks_list = [checkpoint]


# In[47]:


history = model.fit(cc_input_data,cc_input_data_out ,validation_data=(cc_val_data,cc_val_data_out), batch_size=128, epochs=16, callbacks=callbacks_list)

