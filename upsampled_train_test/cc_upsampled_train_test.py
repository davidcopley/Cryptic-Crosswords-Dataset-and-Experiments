import pandas as pd
import os

my_path = os.path.abspath(os.path.dirname(__file__))


def load_upsampled_train_data():
    path = os.path.join(my_path, "./upsampled_train.csv")
    return pd.read_csv(path)


def load_upsampled_train_data():
    path = os.path.join(my_path, "./upsampled_train.csv")
    return pd.read_csv(path).drop_duplicates()


def load_test_data():
    path = os.path.join(my_path, "./test.csv")
    return pd.read_csv(path)
