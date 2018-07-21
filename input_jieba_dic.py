# coding=utf8
import pandas as pd


class InputJiebaDic():
    def __init__(self, dic_file):
        self.df = pd.read_csv(dic_file, header=None)
        self.df.columns = ['word']

    def get_jieba_dic(self):
        return self.df
