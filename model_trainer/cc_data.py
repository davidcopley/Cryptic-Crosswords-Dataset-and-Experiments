import pickle
import os

my_path = os.path.abspath(os.path.dirname(__file__))


def get_cc_data_with_charades():
    path = os.path.join(my_path, "../upsampled_train_val_test/upsampled_data_with_charades.pickle")
    with open(path,'rb') as f:
        data = pickle.load(f)
    return data


def get_cc_data_without_charades():
    path = os.path.join(my_path, "../upsampled_train_val_test/upsampled_data_without_charades.pickle")
    with open(path,'rb') as f:
        data = pickle.load(f)
    return data
