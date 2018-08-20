import tensorflow as tf
import numpy as np
from data.Data_deal import DataDealSeq
from model import embedding, encoder_decoder, loss, encoder, decoder, decoder_1
from nltk import bleu
import os

gpu_id = 2

os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % gpu_id


class Config(object):
    '''
    默认配置
    '''
    learning_rate = 0.005
    num_samples = 5000
    batch_size = 128
    encoder_len = 10
    decoder_len = 15
    embedding_dim = 50
    hidden_dim = 100
    train_dir = '/seq2seq_train_data.txt'
    dev_dir = '/seq2seq_dev_data.txt'
    test_dir = '/test.txt'
    model_dir = './save_model/seq2seq_bilstm_.ckpt'
    train_num = 50
    use_cpu_num = 8
    summary_write_dir = "./tmp/seq2seq_my.log"
    epoch = 1000
    encoder_mod = "lstm"  # Option=[bilstm lstm lstmTF cnn ]
    use_sample = False
    beam_size = 168
    use_MMI = False


config = Config()
tf.app.flags.DEFINE_float("learning_rate", config.learning_rate, "学习率")
tf.app.flags.DEFINE_integer("num_samples", config.num_samples, "采样损失函数的采样的样本数")
tf.app.flags.DEFINE_integer("batch_size", config.batch_size, "批处理的样本数量")
tf.app.flags.DEFINE_integer("encoder_len", config.encoder_len, "编码数据的长度")
tf.app.flags.DEFINE_integer("decoder_len", config.decoder_len, "解码数据的长度")
tf.app.flags.DEFINE_integer("embedding_dim", config.embedding_dim, "词嵌入惟独.")
tf.app.flags.DEFINE_integer("hidden_dim", config.hidden_dim, "中间节点维度.")
tf.app.flags.DEFINE_integer("train_num", config.train_num, "训练次数，每次训练加载上次的模型.")
tf.app.flags.DEFINE_integer("use_cpu_num", config.use_cpu_num, "限定使用cpu的个数")
tf.app.flags.DEFINE_integer("epoch", config.epoch, "每轮训练迭代次数")
tf.app.flags.DEFINE_integer("beam_size", config.beam_size, "束搜索规模")
tf.app.flags.DEFINE_string("summary_write_dir", config.summary_write_dir, "训练数据过程可视化文件保存地址")
tf.app.flags.DEFINE_string("train_dir", config.train_dir, "训练数据的路径")
tf.app.flags.DEFINE_string("dev_dir", config.dev_dir, "验证数据文件路径")
tf.app.flags.DEFINE_string("test_dir", config.test_dir, "测试数据文件路径")
tf.app.flags.DEFINE_string("model_dir", config.model_dir, "模型保存路径")
tf.app.flags.DEFINE_string("encoder_mod", config.encoder_mod, "编码层使用的模型 lstm bilstm cnn")
tf.app.flags.DEFINE_boolean("sample_loss", True, "是否采用采样的损失函数")  # true for prediction
tf.app.flags.DEFINE_string("mod", "train", "默认为训练")  # true for prediction
tf.app.flags.DEFINE_boolean('use_MMI', config.use_MMI, "是否使用最大互信息来增加解码的多样性")
FLAGS = tf.app.flags.FLAGS


class Seq2SeqRl(object):

    def __init__(self, scope, encoder_word_num, decoder_word_num):

        with tf.variable_scope(name_or_scope=scope):
            self.encoder_word_num = encoder_word_num
            self.decoder_word_num = decoder_word_num
            self.__build_model__()

    def __build_model__(self):
        with tf.device('/gpu:%s' % gpu_id):
            self.encoder_input = tf.placeholder(shape=(None, FLAGS.encoder_len), dtype=tf.int32, name='encoder_input')

            self.encoder_seq_len = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_seq_len')

            self.decoder_input = tf.placeholder(shape=(None, FLAGS.decoder_len), dtype=tf.int32, name='decoder_input')

            self.decoder_mask = tf.placeholder(shape=(None, FLAGS.decoder_len), dtype=tf.int32, name='decoder_mask')

            self.decoder_seq_len = tf.placeholder(shape=(None,), dtype=tf.int32, name='decoder_seq_len')

            self.decoder_label = tf.placeholder(shape=(None, FLAGS.decoder_len), dtype=tf.int32, name='decoder_label')

            self.encoder_emb, _ = embedding(self.encoder_input, self.encoder_word_num, FLAGS.embedding_dim,
                                            name='encoder_emb')
            self.decoder_emb, dec_emb = embedding(self.decoder_input, self.decoder_word_num, FLAGS.embedding_dim,
                                                  name='encoder_emb',reuse=True)
            self.decoder_label_emb, _ = embedding(self.decoder_label, self.decoder_word_num, FLAGS.embedding_dim,
                                                  name='encoder_emb', reuse=True)

            self.encoder_cell = tf.contrib.rnn.LSTMCell(FLAGS.hidden_dim,
                                                        initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123),
                                                        state_is_tuple=True)
            self.modd='enc_dec'

            if self.modd=='enc_dec':
                self.outs=encoder_decoder(self.encoder_input,FLAGS.encoder_len,self.decoder_input,FLAGS.decoder_len,self.encoder_cell,FLAGS.hidden_dim,
                                self.encoder_word_num,self.decoder_word_num,name='enc_dec',feed_previous=True)

            else:
                encoder_out, encoder_states, encoder_state_tuple = encoder(self.encoder_emb, FLAGS.encoder_len,
                                                                           self.encoder_seq_len, self.encoder_cell,
                                                                           name='lstm')

                self.outs = decoder_1(encoder_out=encoder_state_tuple, encoder_state=encoder_states,
                                      encoder_seq_len=self.encoder_seq_len, encoder_len=FLAGS.encoder_len,
                                      decoder_emb=self.decoder_emb, decoder_seq_len=self.decoder_seq_len,
                                      decoder_len=FLAGS.decoder_len,
                                      decoder_vocab_num=self.decoder_word_num, cell=self.encoder_cell, name='decoder',
                                      decoder_embedding=dec_emb, hidden_dim=FLAGS.hidden_dim,mod=FLAGS.mod)


            self.soft_logit = tf.nn.log_softmax(self.outs, 2)
            self.preds = tf.to_int32(tf.argmax(self.soft_logit,2))


            self.istarget = tf.to_float(tf.not_equal(self.decoder_label, 0))
            self.acc = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self.decoder_label))*self.istarget)/ (tf.reduce_sum(self.istarget))


            self.loss = loss(self.outs, self.decoder_label, self.decoder_word_num, sample_num=FLAGS.num_samples,
                             decoder_seq_len=self.decoder_seq_len,
                             decoder_len=FLAGS.decoder_len, decoder_mask=self.decoder_mask)
            self.opt = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(self.loss)

            # self.outs=encoder_decoder(encoder_emb=self.encoder_input,decoder_emb=self.decoder_input,encoder_len=FLAGS.encoder_len,
            #                 decoder_len=FLAGS.decoder_len,cell=self.encoder_cell,hidden_dim=FLAGS.hidden_dim,
            #                 encoder_voab_num=self.encoder_word_num,decoer_vocab_num=self.decoder_word_num,feed_previous=True,name='encoder_decoder')
            #
            # self.soft_logit = tf.nn.softmax(self.outs, 1)
            #
            # self.loss=loss(self.outs,self.decoder_label,self.decoder_word_num,sample_num=FLAGS.num_samples,decoder_seq_len=self.decoder_seq_len,
            #                decoder_len=FLAGS.decoder_len,decoder_mask=self.decoder_mask)
            # self.opt=tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(self.loss)

    def __build_rl_model__(self):
        pass

    def train(self, dd):

        config = tf.ConfigProto(allow_soft_placement=True)
        saver = tf.train.Saver()

        with tf.Session(config=config) as sess:
            #saver.restore(sess,'%s'%FLAGS.model_dir)
            sess.run(tf.global_variables_initializer())
            #
            # decoder_input, decoder_label, encoder_input, decoder_len, encoder_len, _ = dd.next_batch()
            # losses,_=sess.run([self.loss,self.opt],feed_dict={self.encoder_input:encoder_input,
            #                                                               self.decoder_input:decoder_input,
            #                                                               self.decoder_label:decoder_label,
            #                                                               self.encoder_seq_len:encoder_len,
            #                                                               self.decoder_seq_len:decoder_len})

            train_data, test_data = dd.get_train_data()
            result_content_input, result_content_decoder, result_content_len_list, result_title, result_title_len_list, result_loss_weight = train_data
            num_batch = int(float(len(result_content_input)) / float(FLAGS.batch_size))

            for i in range(FLAGS.epoch):
                all_loss = 0
                train_acc = 0.0
                for j in range(num_batch):
                    decoder_input = result_content_input[j * FLAGS.batch_size:(j + 1) * FLAGS.batch_size]
                    decoder_label = result_content_decoder[j * FLAGS.batch_size:(j + 1) * FLAGS.batch_size]
                    encoder_input = result_title[j * FLAGS.batch_size:(j + 1) * FLAGS.batch_size]
                    decoder_len = result_content_len_list[j * FLAGS.batch_size:(j + 1) * FLAGS.batch_size]
                    encoder_len = result_title_len_list[j * FLAGS.batch_size:(j + 1) * FLAGS.batch_size]

                    decoder_input = np.zeros_like(decoder_input)
                    decoder_input[:, 0] = 1

                    decoder_mask = np.zeros_like(decoder_input)
                    for index, e in enumerate(decoder_len):
                        decoder_mask[index][:e + 1] = 1
                    losses, soft_logit, _,acc = sess.run([self.loss, self.soft_logit, self.opt,self.acc],
                                                     feed_dict={self.encoder_input: encoder_input,
                                                                self.decoder_input: decoder_input,
                                                                self.decoder_label: decoder_label,
                                                                self.encoder_seq_len: encoder_len,
                                                                self.decoder_seq_len: decoder_len,
                                                                self.decoder_mask: decoder_mask})
                    train_acc += acc
                    all_loss += losses
                #                 all_loss=all_loss/float(num_batch)
                #                 print('this is {} train_loss:{} '.format(i,all_loss))

                test_decoder_input, test_decoder_label, test_decoder_len, \
                test_encoder_input, test_encoder_len, test_result_loss_weight = test_data
                test_decoder_mask = np.zeros_like(test_decoder_input)
                test_decoder_input = np.zeros_like(test_decoder_input)
                test_decoder_input[:, 0] = 1

                for index, e in enumerate(test_decoder_len):
                    test_decoder_mask[index][:e + 1] = 1
                test_batch = 500
                test_num = int(test_decoder_input.shape[0] / test_batch)
                all_test_loss = 0.0
                all_test_acc = 0.0
                for ii in range(test_num):
                    test_encoder_input_batch = test_encoder_input[ii * test_batch:(ii + 1) * test_batch]
                    test_decoder_input_batch = test_decoder_input[ii * test_batch:(ii + 1) * test_batch]
                    test_decoder_label_batch = test_decoder_label[ii * test_batch:(ii + 1) * test_batch]
                    test_encoder_len_batch = test_encoder_len[ii * test_batch:(ii + 1) * test_batch]
                    test_decoder_len_batch = test_decoder_len[ii * test_batch:(ii + 1) * test_batch]
                    test_decoder_mask_batch = test_decoder_mask[ii * test_batch:(ii + 1) * test_batch]


                    test_losses, test_soft_logit,test_acc = sess.run([self.loss, self.soft_logit,self.acc],
                                                            feed_dict={self.encoder_input: test_encoder_input_batch,
                                                                       self.decoder_input: test_decoder_input_batch,
                                                                       self.decoder_label: test_decoder_label_batch,
                                                                       self.encoder_seq_len: test_encoder_len_batch,
                                                                       self.decoder_seq_len: test_decoder_len_batch,
                                                                       self.decoder_mask: test_decoder_mask_batch})
                    all_test_loss += test_losses
                    all_test_acc += test_acc

                all_test_loss = all_test_loss / float(test_batch)
                all_loss = all_loss / float(num_batch)
                all_train_acc = train_acc / float(num_batch)
                all_test_acc = all_test_acc / float(test_batch)
                print(
                    'this is {} train_loss:{} train_acc:{} test_loss:{}  test_acc:{}'.format(i, all_loss, all_train_acc,
                                                                                             all_test_loss,
                                                                                             all_test_acc))
                saver.save(sess, '%s' % FLAGS.model_dir)

    def acc(self,pre, label,mask):

        pre_index = np.argmax(pre, 2)
        acc = float(np.sum(np.equal(pre_index, label))) / float(label.shape[0] * label.shape[1])
        return acc

    def infer(self, dd):
        train_data, test_data = dd.get_train_data()
        result_content_input, result_content_decoder, result_content_len_list, result_title, result_title_len_list, result_loss_weight = train_data

        decoder_input=result_content_input[:100]
        decoder_label=result_content_decoder[:100]
        encoder_input=result_title[:100]
        decoder_len=result_content_len_list[:100]
        encoder_len=result_title_len_list[:100]

        decoder_mask = np.zeros_like(decoder_input)
        for index, e in enumerate(decoder_len):
            decoder_mask[index][:e] = 1
        decoder_input = np.zeros_like(decoder_input)
        decoder_input[:, 0] = 1
        # decoder_label=np.zeros_like(decoder_label)

        saver = tf.train.Saver()
        id2content = dd.id2content
        id2encoder = dd.id2encoder
        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=config) as sess:

            saver.restore(sess, '%s' % FLAGS.model_dir)

            outs, losses = sess.run([self.soft_logit, self.loss], feed_dict={
                self.encoder_input: encoder_input,
                self.decoder_input: decoder_input,
                self.decoder_label: decoder_label,
                self.encoder_seq_len: encoder_len,
                self.decoder_seq_len: decoder_len,
                self.decoder_mask: decoder_mask})
            print(losses)
            all_score = 0.0
            outs = np.argmax(outs, 2)
            for predict, encoder, decoder in zip(outs, encoder_input, decoder_label):
                con = ' '.join([id2encoder[ele] for ele in encoder])
                res = ' '.join([id2content[ele] for ele in predict])
                reference = [id2content[e] for e in decoder]
                socre = bleu([reference], [id2content[ele] for ele in predict])
                all_score += socre
                print(socre, con, '\t\t', res)
                print('\n')
            print('belu socre %s' % (float(all_score) / float(len(encoder_input))))


def main(_):
    dd = DataDealSeq(train_path="/train_sample.txt", test_path="/test.txt",
                     dev_path="/test.txt",
                     dim=FLAGS.hidden_dim, batch_size=FLAGS.batch_size, content_len=FLAGS.decoder_len,
                     title_len=FLAGS.encoder_len, flag="train_new",name='seq2seq')

    decoder_word_num = len(dd.content_vocab)
    encoder_word_num = len(dd.title_vocab)
    ssl = Seq2SeqRl(encoder_word_num=encoder_word_num, decoder_word_num=decoder_word_num, scope='seqRl')

    if FLAGS.mod == 'train':
        ssl.train(dd)
    elif FLAGS.mod == 'infer':
        ssl.infer(dd)


if __name__ == '__main__':
    tf.app.run()