# coding=utf8
#################在PAI平台移除#############################
from input_traindata import InputTrainData
from input_jieba_dic import InputJiebaDic
from input_testdata import InputTestData
##############################################

import jieba
from gensim.models import word2vec
import tensorflow as tf
from tensorflow.contrib import learn
import tensorflow.contrib.slim as slim
import numpy as np

import time
import datetime
import logging
import os

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

    return model


class MyVocabularyProcessor(learn.preprocessing.VocabularyProcessor):
    def __init__(self,
                 max_document_length,
                 min_frequency=0,
                 vocabulary=None):

        def tokenizer_word(iterator):
            for sentence in iterator:
                yield list(jieba_module.lcut(sentence))

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
    for item in train_data['label']:
        flag = int(item)
        lable = [0] * 2
        if flag > 0:
            lable[1] = 1
        else:
            lable[0] = 1
        train_y.append(np.array(lable, dtype='float32'))

    for item in test_data['sent1']:
        x1_text.append(item)
        test_x1.append(item)
    for item in test_data['sent2']:
        x2_text.append(item)
        test_x2.append(item)
    for item in test_data['label']:
        flag = int(item)
        lable = [0] * 2
        if flag > 0:
            lable[1] = 1
        else:
            lable[0] = 1
        test_y.append(np.array(lable, dtype='float32'))

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


# utils
def compute_l1_distance(x, y):
    with tf.name_scope('l1_distance'):
        d = tf.reduce_sum(tf.abs(tf.subtract(x, y)), axis=1)
        return d


def compute_euclidean_distance(x, y):
    with tf.name_scope('euclidean_distance'):
        d = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(x, y)), axis=1))
        return d


def compute_pearson_distance(x, y):
    with tf.name_scope("pearson"):
        mid1 = tf.reduce_mean(x * y, axis=1) - \
               tf.reduce_mean(x, axis=1) * tf.reduce_mean(y, axis=1)
        mid2 = tf.sqrt(tf.reduce_mean(tf.square(x), axis=1) - tf.square(tf.reduce_mean(x, axis=1))) * \
               tf.sqrt(tf.reduce_mean(tf.square(y), axis=1) - tf.square(tf.reduce_mean(y, axis=1)))
        return mid1 / mid2


def compute_cosine_distance(x, y):
    with tf.name_scope('cosine_distance'):
        x_norm = tf.sqrt(tf.reduce_sum(tf.square(x), axis=1))
        y_norm = tf.sqrt(tf.reduce_sum(tf.square(y), axis=1))
        x_y = tf.reduce_sum(tf.multiply(x, y), axis=1)
        d = tf.divide(x_y, tf.multiply(x_norm, y_norm))
        return d


def comU1(x, y):
    result = [compute_cosine_distance(x, y), compute_l1_distance(x, y)]
    # result = [compute_euclidean_distance(x, y), compute_euclidean_distance(x, y), compute_euclidean_distance(x, y)]
    return tf.stack(result, axis=1)


def comU2(x, y):
    # result = [compute_cosine_distance(x, y), compute_euclidean_distance(x, y)]
    # return tf.stack(result, axis=1)
    return tf.expand_dims(compute_cosine_distance(x, y), -1)


class MPCNN():
    def __init__(self, num_classes, embedding_size, filter_sizes, num_filters, n_hidden,
                 input_x1, input_x2, input_y, dropout_keep_prob, l2_reg_lambda):
        def init_weight(shape, name):
            var = tf.Variable(tf.truncated_normal(shape, mean=0, stddev=1.0), name=name)
            return var

        self.embedding_size = embedding_size
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.num_classes = num_classes
        self.poolings = [tf.reduce_max, tf.reduce_min, tf.reduce_mean]

        self.input_x1 = input_x1
        self.input_x2 = input_x2
        self.input_y = input_y
        self.dropout_keep_prob = dropout_keep_prob
        self.l2_loss = tf.constant(0.0)
        self.l2_reg_lambda = l2_reg_lambda
        self.W1 = [init_weight([filter_sizes[0], embedding_size, 1, num_filters[0]], "W1_0"),
                   init_weight([filter_sizes[1], embedding_size, 1, num_filters[0]], "W1_1"),
                   init_weight([filter_sizes[2], embedding_size, 1, num_filters[0]], "W1_2")]
        self.b1 = [tf.Variable(tf.constant(0.1, shape=[num_filters[0]]), "b1_0"),
                   tf.Variable(tf.constant(0.1, shape=[num_filters[0]]), "b1_1"),
                   tf.Variable(tf.constant(0.1, shape=[num_filters[0]]), "b1_2")]

        self.W2 = [init_weight([filter_sizes[0], embedding_size, 1, num_filters[1]], "W2_0"),
                   init_weight([filter_sizes[1], embedding_size, 1, num_filters[1]], "W2_1")]
        self.b2 = [tf.Variable(tf.constant(0.1, shape=[num_filters[1], embedding_size]), "b2_0"),
                   tf.Variable(tf.constant(0.1, shape=[num_filters[1], embedding_size]), "b2_1")]
        self.h = num_filters[0] * len(self.poolings) * 2 + \
                 num_filters[1] * (len(self.poolings) - 1) * (len(filter_sizes) - 1) * 3 + \
                 len(self.poolings) * len(filter_sizes) * len(filter_sizes) * 3

        self.Wh1 = tf.Variable(tf.random_normal([604, n_hidden], stddev=0.01), name='Wh1')
        self.bh1 = tf.Variable(tf.constant(0.1, shape=[n_hidden]), name="bh1")

        n_hidden2 = n_hidden
        self.Wh2 = tf.Variable(tf.random_normal([n_hidden, n_hidden2], stddev=0.01), name='Wh2')
        self.bh2 = tf.Variable(tf.constant(0.1, shape=[n_hidden2]), name="bh2")

        n_hidden3 = n_hidden
        self.Wh3 = tf.Variable(tf.random_normal([n_hidden2, n_hidden3], stddev=0.01), name='Wh3')
        self.bh3 = tf.Variable(tf.constant(0.1, shape=[n_hidden3]), name="bh3")

        self.Wo = tf.Variable(tf.random_normal([n_hidden, num_classes], stddev=0.01), name='Wo')
        self.bo = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="bo")

    def attention(self):
        sent1_unstack = tf.unstack(self.input_x1, axis=1)
        sent2_unstack = tf.unstack(self.input_x2, axis=1)
        D = []
        for i in range(len(sent1_unstack)):
            d = []
            for j in range(len(sent2_unstack)):
                dis = compute_cosine_distance(sent1_unstack[i], sent2_unstack[j])
                # dis:[batch_size, 1(channels)]
                d.append(dis)
            D.append(d)
            print(i)
        D = tf.reshape(D, [-1, len(sent1_unstack), len(sent2_unstack), 1])
        A = [tf.nn.softmax(tf.expand_dims(tf.reduce_sum(D, axis=i), 2)) for i in [2, 1]]
        atten_embed = []
        atten_embed.append(tf.concat([self.input_x1, A[0] * self.input_x1], 2))
        atten_embed.append(tf.concat([self.input_x2, A[1] * self.input_x2], 2))
        return atten_embed

    def per_dim_conv_layer(self, x, w, b, pooling):
        '''

        :param input: [batch_size, sentence_length, embed_size, 1]
        :param w: [ws, embedding_size, 1, num_filters]
        :param b: [num_filters, embedding_size]
        :param pooling:
        :return:
        '''
        # unpcak the input in the dim of embed_dim
        input_unstack = tf.unstack(x, axis=2)
        w_unstack = tf.unstack(w, axis=1)
        b_unstack = tf.unstack(b, axis=1)
        convs = []
        for i in range(x.get_shape()[2]):
            conv = tf.nn.conv1d(input_unstack[i], w_unstack[i], stride=1, padding="VALID")
            conv = slim.batch_norm(inputs=conv, activation_fn=tf.nn.tanh, is_training=self.is_training)
            convs.append(conv)
        conv = tf.stack(convs, axis=2)
        pool = pooling(conv, axis=1)

        return pool

    def bulit_block_A(self, x):
        # bulid block A and cal the similarity according to algorithm 1
        out = []
        with tf.name_scope("bulid_block_A"):
            for pooling in self.poolings:
                pools = []
                for i, ws in enumerate(self.filter_sizes):
                    with tf.name_scope("conv-pool-%s" % ws):
                        # print ('==========x==========')
                        # print (x)
                        # exit(0)

                        conv = tf.nn.conv2d(x, self.W1[i], strides=[1, 1, 1, 1], padding="VALID")
                        conv = slim.batch_norm(inputs=conv, activation_fn=tf.nn.tanh, is_training=self.is_training)
                        pool = pooling(conv, axis=1)
                    pools.append(pool)
                out.append(pools)
            return out

    def bulid_block_B(self, x):
        out = []
        with tf.name_scope("bulid_block_B"):
            for pooling in self.poolings[:-1]:
                pools = []
                with tf.name_scope("conv-pool"):
                    for i, ws in enumerate(self.filter_sizes[:-1]):
                        with tf.name_scope("per_conv-pool-%s" % ws):
                            pool = self.per_dim_conv_layer(x, self.W2[i], self.b2[i], pooling)
                        pools.append(pool)
                    out.append(pools)
            return out

    def similarity_sentence_layer(self):
        # atten = self.attention() #[batch_size, length, 2*embedding, 1]
        sent1 = self.bulit_block_A(self.input_x1)
        sent2 = self.bulit_block_A(self.input_x2)
        fea_h = []
        with tf.name_scope("cal_dis_with_alg1"):
            for i in range(3):
                regM1 = tf.concat(sent1[i], 1)
                regM2 = tf.concat(sent2[i], 1)
                for k in range(self.num_filters[0]):
                    fea_h.append(comU2(regM1[:, :, k], regM2[:, :, k]))

        # self.fea_h = fea_h

        fea_a = []
        with tf.name_scope("cal_dis_with_alg2_2-9"):
            for i in range(3):
                for j in range(len(self.filter_sizes)):
                    for k in range(len(self.filter_sizes)):
                        fea_a.append(comU1(sent1[i][j][:, 0, :], sent2[i][k][:, 0, :]))
        #
        sent1 = self.bulid_block_B(self.input_x1)
        sent2 = self.bulid_block_B(self.input_x2)

        fea_b = []
        with tf.name_scope("cal_dis_with_alg2_last"):
            for i in range(len(self.poolings) - 1):
                for j in range(len(self.filter_sizes) - 1):
                    for k in range(self.num_filters[1]):
                        fea_b.append(comU1(sent1[i][j][:, :, k], sent2[i][j][:, :, k]))
        # self.fea_b = fea_b
        return tf.concat(fea_h + fea_a + fea_b, 1)

    def similarity_measure_layer(self, is_training=True):
        self.is_training = is_training
        fea = self.similarity_sentence_layer()
        # fea = tf.nn.dropout(fea, self.dropout_keep_prob)

        with tf.name_scope("full_connect_layer"):
            h1 = tf.nn.tanh(tf.matmul(fea, self.Wh1) + self.bh1)
            h1 = tf.nn.dropout(h1, self.dropout_keep_prob)

            # h2 = tf.nn.tanh(tf.matmul(h1, self.Wh2) + self.bh2)
            # h2 = tf.nn.dropout(h2, self.dropout_keep_prob)
            #
            # h3 = tf.nn.tanh(tf.matmul(h2, self.Wh3) + self.bh3)
            # h3 = tf.nn.dropout(h3, self.dropout_keep_prob)

            h3 = h1

            self.scores = tf.matmul(h3, self.Wo) + self.bo
            self.output = tf.nn.softmax(self.scores)

        # CalculateMean cross-entropy loss
        reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-4), tf.trainable_variables())
        with tf.name_scope("loss"):
            # self.loss = -tf.reduce_sum(self.input_y * tf.log(self.output))
            self.loss = tf.reduce_sum(tf.square(tf.subtract(self.input_y, self.output))) + reg

            # self.loss = tf.reduce_mean(
            #     tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y))
            # self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * self.l2_loss

        with tf.name_scope("accuracy"):
            self.input_y_vector = tf.argmax(self.input_y, 1)
            self.output_y_vector = tf.argmax(self.output, 1, name="temp_sim")

            # self.accuracy = tf.reduce_mean(
            #     tf.cast(tf.equal(tf.argmax(self.input_y, 1), tf.argmax(self.scores, 1)), tf.float32))
            self.accuracy = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(self.input_y, 1), tf.argmax(self.output, 1)), tf.float32))

        with tf.name_scope('f1'):
            ones_like_actuals = tf.ones_like(self.input_y_vector)
            zeros_like_actuals = tf.zeros_like(self.input_y_vector)
            ones_like_predictions = tf.ones_like(self.output_y_vector)
            zeros_like_predictions = tf.zeros_like(self.output_y_vector)

            tp = tf.reduce_sum(
                tf.cast(
                    tf.logical_and(
                        tf.equal(self.input_y_vector, ones_like_actuals),
                        tf.equal(self.output_y_vector, ones_like_predictions)
                    ),
                    'float'
                )
            )

            tn = tf.reduce_sum(
                tf.cast(
                    tf.logical_and(
                        tf.equal(self.input_y_vector, zeros_like_actuals),
                        tf.equal(self.output_y_vector, zeros_like_predictions)
                    ),
                    'float'
                )
            )

            fp = tf.reduce_sum(
                tf.cast(
                    tf.logical_and(
                        tf.equal(self.input_y_vector, zeros_like_actuals),
                        tf.equal(self.output_y_vector, ones_like_predictions)
                    ),
                    'float'
                )
            )

            fn = tf.reduce_sum(
                tf.cast(
                    tf.logical_and(
                        tf.equal(self.input_y_vector, ones_like_actuals),
                        tf.equal(self.output_y_vector, zeros_like_predictions)
                    ),
                    'float'
                )
            )

            precision = tp / (tp + fp)
            recall = tp / (tp + fn)

            self.f1 = 2 * precision * recall / (precision + recall)


def main():
    input_train_data = InputTrainData('./train_data/atec_nlp_sim_train.csv')
    input_test_data = InputTestData('./train_data/atec_nlp_sim_test.csv')
    input_jieba_dic = InputJiebaDic('./train_data/dict.txt')

    df_train_data = input_train_data.get_train_data()
    df_test_data = input_test_data.get_test_data()
    df_jieba_dic = input_jieba_dic.get_jieba_dic()
    #########################以上在PAI平台移除#################################################33
    # 创建一个logger
    logger = logging.getLogger('mylogger')
    logger.setLevel(logging.DEBUG)

    # 创建一个handler，用于写入日志文件
    timestamp = str(int(time.time()))

    # 再创建一个handler，用于输出到控制台
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # 定义handler的输出格式
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] ## %(message)s')
    ch.setFormatter(formatter)

    # 给logger添加handler
    logger.addHandler(ch)

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
                # print(index,w, arr)
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

        setence_model = MPCNN(NUM_CLASSES, EMBEDDING_DIM, filter_size,
                              [NUM_FILTERS_A, NUM_FILTER_B], N_HIDDEN,
                              input_x1, input_x2, input_y, dropout_keep_prob, L2_REG_LAMBDA)

        global_step = tf.Variable(0, name='global_step', trainable=False)
        setence_model.similarity_measure_layer()
        optimizer = tf.train.AdamOptimizer(LR)
        grads_and_vars = optimizer.compute_gradients(setence_model.loss)
        train_step = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        # print("Writing to {}\n".format(out_dir))
        #
        loss_summary = tf.summary.scalar("loss", setence_model.loss)
        acc_summary = tf.summary.scalar("accuracy", setence_model.accuracy)
        f1_summary = tf.summary.scalar('f1', setence_model.f1)
        #
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, f1_summary])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
        #
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary, f1_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)


        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables())

        # Write vocabulary
        # vocab_processor.save(os.path.join(checkpoint_dir, "vocab"))

        def train(x1_batch, x2_batch, y_batch):
            feed_dict = {
                input_1: x1_batch,
                input_2: x2_batch,
                input_3: y_batch,
                dropout_keep_prob: FULL_CONNECT_LAYER_DROPOUT
            }
            _, step, summaries, batch_loss, accuracy, f1, y_out = sess.run(
                [train_step, global_step, train_summary_op, setence_model.loss, setence_model.accuracy,
                 setence_model.f1,
                 setence_model.output],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            logger.info(
                "{}: step {}, loss {:g}, acc {:g}, f1 {:g}".format(time_str, step, batch_loss, accuracy, f1))
            # logger.info('y_out= {}'.format(y_out))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x1_batch, x2_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
                input_1: x1_batch,
                input_2: x2_batch,
                input_3: y_batch,
                dropout_keep_prob: 1
            }
            step, summaries, batch_loss, accuracy, f1 = sess.run(
                [global_step, dev_summary_op, setence_model.loss, setence_model.accuracy, setence_model.f1],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            dev_summary_writer.add_summary(summaries, step)
            # if writer:
            #     writer.add_summary(summaries, step)

            return batch_loss, accuracy

        def batch_iter(data, batch_size, num_epochs, shuffle=True):
            data = np.array(data)
            data_size = len(data)
            num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
            for epoch in range(num_epochs):
                # Shuffle the data at each epoch
                if shuffle:
                    shuffle_indices = np.random.permutation(np.arange(data_size))
                    shuffled_data = data[shuffle_indices]
                else:
                    shuffled_data = data
                for batch_num in range(num_batches_per_epoch):
                    start_index = batch_num * batch_size
                    end_index = min((batch_num + 1) * batch_size, data_size)
                    yield shuffled_data[start_index:end_index]

        sess.run(tf.global_variables_initializer())
        batches = batch_iter(list(zip(x_train[0], x_train[1], y_train)), BATCH_SIZE, NUM_EPOCHS)

        for batch in batches:
            x1_batch, x2_batch, y_batch = zip(*batch)
            train(x1_batch, x2_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % EVALUATE_EVERY == 0:
                total_dev_loss = 0.0
                total_dev_accuracy = 0.0

                logger.info("\nEvaluation:")
                dev_batches = batch_iter(list(zip(x_test[0], x_test[1], y_test)), BATCH_SIZE, 1)
                for dev_batch in dev_batches:
                    x1_dev_batch, x2_dev_batch, y_dev_batch = zip(*dev_batch)
                    dev_loss, dev_accuracy = dev_step(x1_dev_batch, x2_dev_batch, y_dev_batch)
                    total_dev_loss += dev_loss
                    total_dev_accuracy += dev_accuracy
                total_dev_accuracy = total_dev_accuracy / (len(y_test) / BATCH_SIZE)
                logger.info(
                    "dev_loss {:g}, dev_acc {:g}, num_dev_batches {:g}".format(total_dev_loss, total_dev_accuracy,
                                                                               len(y_test) / BATCH_SIZE))
                # train_summary_writer.add_summary(summaries)

            if current_step % CHECKPOINT_EVERY == 0:
                saver.save(sess, checkpoint_prefix, global_step=current_step)

        logger.info("Optimization Finished!")


if __name__ == '__main__':
    main()
