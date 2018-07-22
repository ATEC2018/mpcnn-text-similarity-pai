# coding=utf8
import pandas as pd


class InputTrainData():
    def __init__(self, train_file):
        self.df = pd.read_csv(train_file, header=None, sep='\t')
        self.df.columns = ['id', 'sent1', 'sent2', 'label']

    def get_train_data(self):
        return self.df
