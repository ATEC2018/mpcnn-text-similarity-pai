# coding=utf8
from input_traindata import InputTrainData
from input_jieba_dic import InputJiebaDic
from input_testdata import InputTestData

import jieba
from gensim.models import word2vec
import tensorflow as tf
from tensorflow.contrib import learn
import numpy as np

# 词向量维数
EMBEDDING_DIM = 128
# block A filter个数(model)
NUM_FILTERS_A = 50
# block B filter个数(model)
NUM_FILTER_B = 50
# 全连接层中隐藏层的单元个数
N_HIDDEN = 150
# 句子最多包含单词数(词)
SENTENCE_LENGTH = 40
# 结果分类个数(二分类后面会使用sigmod 进行优化)
NUM_CLASSES = 2
# L2正规化系数
L2_REG_LAMBDA = 1
# 训练epoch个数
NUM_EPOCHS = 8500
# mini batch大小
BATCH_SIZE = 64
# 评估周期(单位step)
EVALUATE_EVERY = 100
# 模型存档周期
CHECKPOINT_EVERY = 2000
# 优化器学习率
LR = 1e-3

# llow device soft device placement
ALLOW_SOFT_PLACEMENT = True
# Log placement of ops on devices
LOG_DEVICE_PLACEMENT = False

# 卷积filter大小
filter_size = [1, 2, SENTENCE_LENGTH]

# 全连接层Dropout
FULL_CONNECT_LAYER_DROPOUT = 0.8

# 词向量训练相关
MIN_COUNT = 1

# 放在这里很丑陋（暂时先这样吧）
jieba_module = jieba


def jieba_init(module, dic):
    for word in dic['word']:
        module.add_word(word)


def train_word2vec(train_data):
    sentences = []

    for item in train_data['sent1']:
        sentences.append(item)

    for item in train_data['sent2']:
        sentences.append(item)

    corpus = []

    for sten in sentences:
        corpus.append(list(jieba_module.lcut(sten)))

    model = word2vec.Word2Vec(corpus, size=EMBEDDING_DIM, min_count=MIN_COUNT)
    model.init_sims(replace=True)

    print(type(model))
    print(type(model.wv))

    return model


def tokenizer_word(iterator):
    for sentence in iterator:
        yield list(jieba_module.lcut(sentence))


class MyVocabularyProcessor(learn.preprocessing.VocabularyProcessor):
    def __init__(self,
                 max_document_length,
                 min_frequency=0,
                 vocabulary=None):

        tokenizer_fn = tokenizer_word
        self.sup = super(MyVocabularyProcessor, self)
        self.sup.__init__(max_document_length, min_frequency, vocabulary, tokenizer_fn)

    def transform(self, raw_documents):
        for tokens in self._tokenizer(raw_documents):
            word_ids = np.zeros(self.max_document_length, np.int64)
            for idx, token in enumerate(tokens):
                if idx >= self.max_document_length:
                    break
                word_ids[idx] = self.vocabulary_.get(token)
            yield word_ids


def get_dataset(train_data, test_data, sentence_len, batch_size):
    x1_text = []
    x2_text = []

    train_x1 = []
    train_x2 = []
    train_y = []
    test_x1 = []
    test_x2 = []
    test_y = []

    for item in train_data['sent1']:
        x1_text.append(item)
        train_x1.append(item)
    for item in train_data['sent2']:
        x2_text.append(item)
        train_x2.append(item)

    for item in test_data['sent1']:
        x1_text.append(item)
        test_x1.append(item)
    for item in test_data['sent2']:
        x2_text.append(item)
        test_x2.append(item)

    x1_text = np.asarray(x1_text)
    x2_text = np.asarray(x2_text)

    vocab_processor = MyVocabularyProcessor(sentence_len, min_frequency=0)
    vocab_processor.fit_transform(np.concatenate((x2_text, x1_text), axis=0))

    train_x1 = np.asarray(train_x1)
    train_x2 = np.asarray(train_x2)
    train_y = np.asarray(train_y)
    test_x1 = np.asarray(test_x1)
    test_x2 = np.asarray(test_x2)
    test_y = np.asarray(test_y)

    train_x1 = np.asarray(list(vocab_processor.transform(train_x1)))
    train_x2 = np.asarray(list(vocab_processor.transform(train_x2)))
    test_x1 = np.asarray(list(vocab_processor.transform(test_x1)))
    test_x2 = np.asarray(list(vocab_processor.transform(test_x2)))

    num_of_batches = len(train_y) // batch_size

    return [train_x1, train_x2], train_y, [test_x1, test_x2], test_y, vocab_processor, num_of_batches


def main():
    input_train_data = InputTrainData('./train_data/atec_nlp_sim_train.csv')
    input_test_data = InputTestData('./train_data/atec_nlp_sim_test.csv')
    input_jieba_dic = InputJiebaDic('./train_data/dict.txt')

    df_train_data = input_train_data.get_train_data()
    df_test_data = input_test_data.get_test_data()
    df_jieba_dic = input_jieba_dic.get_jieba_dic()

    jieba_init(jieba_module, df_jieba_dic)

    word2vec_model = train_word2vec(df_train_data)

    x_train, y_train, x_test, y_test, vocab_processor, num_of_batches = get_dataset(df_train_data,
                                                                                    df_test_data,
                                                                                    SENTENCE_LENGTH,
                                                                                    BATCH_SIZE)

    with tf.Session() as sess:
        input_1 = tf.placeholder(tf.int32, [None, SENTENCE_LENGTH], name="input_x1")
        input_2 = tf.placeholder(tf.int32, [None, SENTENCE_LENGTH], name="input_x2")
        input_3 = tf.placeholder(tf.float32, [None, NUM_CLASSES], name="input_y")
        dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        init_w = np.random.uniform(-0.25,
                                   0.25,
                                   (len(vocab_processor.vocabulary_), EMBEDDING_DIM)).astype(np.float32)

        for index, w in enumerate(vocab_processor.vocabulary_._mapping):
            arr = []
            if w in word2vec_model.wv:
                arr = word2vec_model.wv[w]
                idx = vocab_processor.vocabulary_.get(w)
                init_w[idx] = np.asarray(arr).astype(np.float32)
            else:
                pass

        with tf.name_scope("embendding"):
            s0_embed = tf.nn.embedding_lookup(init_w, input_1)
            s1_embed = tf.nn.embedding_lookup(init_w, input_2)

        with tf.name_scope("reshape"):
            input_x1 = tf.reshape(s0_embed, [-1, SENTENCE_LENGTH, EMBEDDING_DIM, 1])
            input_x2 = tf.reshape(s1_embed, [-1, SENTENCE_LENGTH, EMBEDDING_DIM, 1])
            input_y = tf.reshape(input_3, [-1, NUM_CLASSES])


if __name__ == '__main__':
    main()
