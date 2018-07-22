# coding=utf8
import pandas as pd


class InputTestData():
    def __init__(self, test_file):
        self.df = pd.read_csv(test_file, header=None, sep='\t')
        self.df.columns = ['id', 'sent1', 'sent2', 'label']

    def get_test_data(self):
        return self.df
