import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import os

my_path = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(my_path, "./pure_cc_clues.pickle"), 'rb') as f:
    dfs = pickle.load(f)


def split_dfs(dfs, shuffle=False):
    train_test = [train_test_split(df, shuffle=shuffle, test_size=0.2) for df in dfs]
    train = [el[0] for el in train_test]
    test = [el[1] for el in train_test]
    return train, test


def get_dataset():
    inpt, test = split_dfs(dfs)
    train, val = split_dfs(inpt, shuffle=True)

    train = pd.concat(train)
    val = pd.concat(val)
    test = pd.concat(test)

    train = train.drop(['is_charade', 'is_lit'], axis=1)
    val = val.drop(['is_charade', 'is_lit'], axis=1)
    test = test.drop(['is_charade', 'is_lit'], axis=1)

    train_x = train.clue
    val_x = val.clue
    test_x = test.clue

    train_y = train[train.columns[4:]]
    val_y = val[train.columns[4:]]
    test_y = test[test.columns[4:]]

    tokenizer = Tokenizer(filters="#$%&()*+/:;<=>@[\]^_`{|}~")
    tokenizer.fit_on_texts(pd.concat([train, val, test]).clue.tolist())

    train_x = pad_sequences(tokenizer.texts_to_sequences(train_x), 15)
    val_x = pad_sequences(tokenizer.texts_to_sequences(val_x), 15)
    test_x = pad_sequences(tokenizer.texts_to_sequences(test_x), 15)
    return (train_x, train_y), (val_x, val_y), (test_x, test_y), tokenizer


def get_upsampled_dataset():
    inpt, test = split_dfs(dfs)
    train, val = split_dfs(inpt, shuffle=True)

    train = pd.concat(train)
    val = pd.concat(val)
    test = pd.concat(test)

    train = train.drop(['is_charade', 'is_lit'], axis=1)
    val = val.drop(['is_charade', 'is_lit'], axis=1)
    test = test.drop(['is_charade', 'is_lit'], axis=1)

    train_y = train[train.columns[4:]]
    val_y = val[train.columns[4:]]
    test_y = test[test.columns[4:]]

    # upsampling
    upsampled_train = train
    categories = np.argmax(train_y.values * 1, axis=1)
    upsampled_train['category'] = categories

    max_size = upsampled_train.groupby('category').count().max()[0]
    tokenizer = Tokenizer(filters="#$%&()*+/:;<=>@[\]^_`{|}~")
    tokenizer.fit_on_texts(pd.concat([train, val, test]).clue.tolist())

    lst = [upsampled_train]
    for class_index, group in upsampled_train.groupby('category'):
        sample = group.sample(max_size - len(group), replace=True, )
        lst.append(sample)
    upsampled_train = pd.concat(lst)
    upsampled_train = upsampled_train.drop('category',axis=1)

    # upsampling

    upsampled_train_x = upsampled_train.clue
    val_x = val.clue
    test_x = test.clue

    upsampled_train_y = upsampled_train[upsampled_train.columns[4:]]

    upsampled_train_x = pad_sequences(tokenizer.texts_to_sequences(upsampled_train_x), 15)
    val_x = pad_sequences(tokenizer.texts_to_sequences(val_x), 15)
    test_x = pad_sequences(tokenizer.texts_to_sequences(test_x), 15)
    return (upsampled_train_x, upsampled_train_y), (val_x, val_y), (test_x, test_y), tokenizer

def get_raw_dataset():
    inpt, test = split_dfs(dfs)
    train, val = split_dfs(inpt, shuffle=True)

    train = pd.concat(train)
    val = pd.concat(val)
    test = pd.concat(test)

    train = train.drop(['is_charade', 'is_lit'], axis=1)
    val = val.drop(['is_charade', 'is_lit'], axis=1)
    test = test.drop(['is_charade', 'is_lit'], axis=1)

    return train,val,test

def get_upsampled_raw_dataset():
    inpt, test = split_dfs(dfs)
    train, val = split_dfs(inpt, shuffle=True)

    train = pd.concat(train)
    val = pd.concat(val)
    test = pd.concat(test)

    train = train.drop(['is_charade', 'is_lit'], axis=1)
    val = val.drop(['is_charade', 'is_lit'], axis=1)
    test = test.drop(['is_charade', 'is_lit'], axis=1)

    train_y = train[train.columns[4:]]

    # upsampling
    upsampled_train = train
    categories = np.argmax(train_y.values * 1, axis=1)
    upsampled_train['category'] = categories

    max_size = upsampled_train.groupby('category').count().max()[0]
    tokenizer = Tokenizer(filters="#$%&()*+/:;<=>@[\]^_`{|}~")
    tokenizer.fit_on_texts(pd.concat([train, val, test]).clue.tolist())

    lst = [upsampled_train]
    for class_index, group in upsampled_train.groupby('category'):
        sample = group.sample(max_size - len(group), replace=True, )
        lst.append(sample)
    upsampled_train = pd.concat(lst)
    upsampled_train = upsampled_train.drop('category',axis=1)

    # upsampling

    return upsampled_train,val,test


