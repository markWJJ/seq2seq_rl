'''
普通的seq2seq 问答模型
'''

import  tensorflow as tf
import sys
import os
sys.path.append("../")
from data import Data_deal
import numpy as np
import logging
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# from seq2seqTF_ops import embedding_encoder,embedding_attention_decoder,embedding_attention_seq2seq,attention_decoder
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filemode='w')
_logger=logging.getLogger("seq2seq")

class Config(object):
    '''
    默认配置
    '''
    learning_rate=0.01
    num_samples=5000
    batch_size=128
    encoder_len=30
    decoder_len=30
    embedding_dim=50
    hidden_dim=100
    train_dir='/seq2seq_train_data.txt'
    dev_dir='/seq2seq_dev_data.txt'
    test_dir='/test.txt'
    model_dir='./save_model/seq2seq_bilstm_.ckpt'
    train_num=50
    use_cpu_num=8
    summary_write_dir="./tmp/seq2seq_my.log"
    epoch=1000
    encoder_mod="lstm" # Option=[bilstm lstm lstmTF cnn ]
    use_sample=False
    beam_size=168
    use_MMI=False

config=Config()
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
tf.app.flags.DEFINE_boolean("sample_loss", True, "是否采用采样的损失函数") # true for prediction
tf.app.flags.DEFINE_string("mod", "train", "默认为训练") # true for prediction
tf.app.flags.DEFINE_boolean('use_MMI',config.use_MMI,"是否使用最大互信息来增加解码的多样性")
FLAGS = tf.app.flags.FLAGS


class Seq2Seq(object):

    def __init__(self,hidden_dim,init_dim,content_len,title_len,con_vocab_len,ti_vocab_len,batch_size):
        self.hidden_dim=hidden_dim
        self.init_dim=init_dim
        self.content_len=content_len
        self.title_len=title_len
        self.content_vocab_len=con_vocab_len
        self.title_vocab_len=ti_vocab_len
        self.batch_size=batch_size
        self.num_class=con_vocab_len

        self.content_input=tf.placeholder(dtype=tf.int32,shape=(None,self.content_len),name='content_input')
        self.content_decoder=tf.placeholder(dtype=tf.int32,shape=(None,self.content_len),name='content_decoder')
        self.mod="beam_decoder"
        self.title=tf.placeholder(dtype=tf.int32,shape=(None,self.title_len),name='title')
        self.content_seq_vec=tf.placeholder(dtype=tf.int32,shape=(None,),name='content_seq_vec')
        self.title_seq_vec=tf.placeholder(dtype=tf.int32,shape=(None,),name='title_seq_vec')
        tf.add_to_collection("input_placehold",self.content_input)
        tf.add_to_collection("input_placehold",self.content_decoder)
        tf.add_to_collection("input_placehold",self.title)
        tf.add_to_collection("input_placehold",self.content_seq_vec)
        tf.add_to_collection("input_placehold",self.title_seq_vec)

        with tf.device('/gpu:1'):

            self.init_loss=9999
            self.best_loss=9999
            self.best_iter=0
            #构建encoder层词向量矩阵变量
            self.embeding_content=tf.Variable(tf.random_uniform(shape=(self.content_vocab_len,self.init_dim),
                                                        dtype=tf.float32),name="embedding_content")
            # 构建decoder层词向量矩阵变量
            self.embeding_title=tf.Variable(tf.random_uniform(shape=(self.title_vocab_len,self.init_dim),
                                                        dtype=tf.float32),name="embedding_title")
            tf.add_to_collection("embedding",self.embeding_content)
            tf.add_to_collection("embedding",self.embeding_title)

            # 定义lstm 单元
            if FLAGS.encoder_mod in ["lstm",'lstmTF']:
                self.cell = tf.contrib.rnn.LSTMCell(self.hidden_dim,
                                                 initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123),
                                                 state_is_tuple=True)
                tf.add_to_collection('lstm_cell',self.cell)
            elif FLAGS.encoder_mod in ['bilstm']:
                self.cell_f = tf.contrib.rnn.LSTMCell(self.hidden_dim,
                                                 initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123),
                                                 state_is_tuple=True)
                self.cell_b = tf.contrib.rnn.LSTMCell(self.hidden_dim,
                                                 initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123),
                                                 state_is_tuple=True)

                self.cell_decoder = tf.contrib.rnn.LSTMCell(2*self.hidden_dim,
                                                 initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123),
                                                 state_is_tuple=True)

            self.content_emb_input=tf.nn.embedding_lookup(self.embeding_content,self.content_input)
            self.content_emb_decoder=tf.nn.embedding_lookup(self.embeding_content,self.content_decoder)
            self.title_emb=tf.nn.embedding_lookup(self.embeding_title,self.title)

            # self.Encoder_Decoder()

            # encoder_outs,encoder_state_c,encoder_state_h=self.Encoder()
            # tf.add_to_collection('encoder_output',encoder_outs)
            # tf.add_to_collection('encoder_state_c',encoder_state_c)
            # tf.add_to_collection('encoder_state_h',encoder_state_h)
            #
            # self.encoder_outs=encoder_outs
            # self.encoder_state_c=encoder_state_c
            # self.encoder_state_h=encoder_state_h
            #
            #
            # decoder_out,decoder_state=self.Decoder(encoder_outs,encoder_state_c,encoder_state_h)
            decoder_out=self.Encoder_Decoder()
            self.loss=self.Loss(decoder_out,self.content_decoder)
            tf.summary.scalar("loss_my",self.loss)

            self.opt=tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(self.loss)

            self.merge_summary=tf.summary.merge_all()
            # optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
            # grads_vars = optimizer.compute_gradients(self.loss)
            # capped_grads_vars = [[tf.clip_by_value(g, -5.0, 5.0), v] for g, v in grads_vars]
            # self.opt = optimizer.apply_gradients(capped_grads_vars)

    def beam_decoder(self):
        # 解码阶段
        if FLAGS.encoder_mod == 'lstm':
            cell=self.cell
        elif FLAGS.encoder_mod == 'bilstm':
            cell=self.cell_decoder
        self.beam_state_c = tf.placeholder(shape=(None, self.hidden_dim), dtype=tf.float32)
        self.beam_state_h = tf.placeholder(shape=(None, self.hidden_dim), dtype=tf.float32)
        self.beam_inputs = tf.placeholder(shape=(None,), dtype=tf.int32)
        self.beam_encoder=tf.placeholder(shape=(None,self.title_len,self.hidden_dim),dtype=tf.float32)
        emb_beam_decoder = tf.nn.embedding_lookup(self.embeding_content, self.beam_inputs)
        beam_state=(self.beam_state_c,self.beam_state_h)
        tf.get_variable_scope().reuse_variables()
        outs, states = tf.contrib.legacy_seq2seq.attention_decoder(
            decoder_inputs=[emb_beam_decoder],
            initial_state=beam_state,
            attention_states=self.beam_encoder,
            cell=cell,
        )
        outs = tf.stack(outs, 1)
        ll = tf.einsum('ijk,kl->ijl', outs, tf.transpose(self.w))
        beam_softmax = tf.nn.softmax(tf.add(ll, self.b))
        self.state = states
        self.beam_softmax = beam_softmax




    def Encoder_Decoder(self):
        '''
        编码+解码
        :return: 
        '''
        encoder_inputs = tf.unstack(self.title, self.title_len, 1)
        decoder_inputs = tf.unstack(self.content_input, self.content_len, 1)
        out_w = tf.Variable(tf.random_uniform(shape=(self.hidden_dim, self.content_vocab_len), maxval=1.0, minval=-1.0),
                            dtype=tf.float32)
        out_b = tf.Variable(tf.random_uniform(shape=(self.content_vocab_len,)), dtype=tf.float32)
        if FLAGS.mod=='train':

            outs,state=tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                encoder_inputs,
                decoder_inputs,
                self.cell,
                num_encoder_symbols=self.title_vocab_len,
                num_decoder_symbols=self.content_vocab_len,
                embedding_size=self.hidden_dim,
                output_projection=(out_w,out_b),
                feed_previous=True,
                dtype=tf.float32)
            # self.loss=self.Loss(outs,self.content_decoder)
            # self.opt=tf.train.AdamOptimizer(0.03).minimize(self.loss)
            return outs
        else:
            outs, state = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                encoder_inputs,
                decoder_inputs,
                self.cell,
                num_encoder_symbols=self.title_vocab_len,
                num_decoder_symbols=self.content_vocab_len,
                embedding_size=self.hidden_dim,
                output_projection=(out_w, out_b),
                feed_previous=True,
                dtype=tf.float32)
            # self.loss=self.Loss(outs,self.content_decoder)
            # self.opt=tf.train.AdamOptimizer(0.03).minimize(self.loss)
            return outs

    def Encoder(self):
        '''
        编码层
        :return: 
        '''
        lstm_input=tf.unstack(self.title_emb,self.title_len,1)
        if FLAGS.encoder_mod=="bilstm":

            (out, fw_state, bw_state) = tf.contrib.rnn.static_bidirectional_rnn(self.cell_f, self.cell_b, lstm_input,
                                                                         dtype=tf.float32,
                                                                         sequence_length=self.title_seq_vec)

            encoder_state_c=tf.concat((fw_state[0],bw_state[0]),1)
            encoder_state_h=tf.concat((fw_state[1],bw_state[1]),1)
            outs=tf.stack(out)
            outs=tf.reshape(outs,[-1,self.title_len,2*self.hidden_dim])

        elif FLAGS.encoder_mod=="lstm":

            out, state = tf.contrib.rnn.static_rnn(self.cell, lstm_input,
                                                    dtype=tf.float32,
                                                    sequence_length=self.title_seq_vec)
            top_states = [
                tf.reshape(e, [-1, 1, self.hidden_dim]) for e in out
            ]
            outs = tf.concat(top_states, 1)
            encoder_state_c=state[0]
            encoder_state_h=state[1]
        elif FLAGS.encoder_mod=="cnn":
            # convd=[height,width,in_channels,out_channels]
            # 第一层卷积层的size [4,embedding_dim,1,10]
            convd_w=tf.Variable(tf.random_uniform(shape=(4,self.init_dim,1,20),minval=-0.1,maxval=0.1),dtype=tf.float32)
            convd_b=tf.Variable(tf.random_uniform(shape=(20,),dtype=tf.float32))
            #strides 一个长为4的list. 表示每次卷积以后在input中滑动的距离 strides.shape=inputs.shape [batch_size,height,width,channels]
            #padding 有SAME和VALID两种选项，表示是否要保留不完全卷积的部分。如果是SAME，则保留
            # 转换shape cnn层的标准输入：[batch_size,height,width,channels]
            cnn_input=tf.reshape(self.title_emb,[-1,self.title_len,self.init_dim,1])
            convd=tf.nn.conv2d(cnn_input,convd_w,strides=[1,1,1,1],padding="SAME") #若滑动stride为1 代表输出维度和输入一致
            convd_1=tf.nn.relu(tf.add(convd,convd_b)) #[batch_size,self.title_len,self.init_dim,out_channels]
            convd_pool_1=tf.nn.max_pool(convd_1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME") # size=[batch_size,title_len/2,init_din/2,20]
            # 第二层 cnn
            convd_w_2=tf.Variable(tf.random_uniform(shape=(4,2,20,32),minval=-0.1,maxval=0.1),dtype=tf.float32)
            convd_b_2=tf.Variable(tf.random_uniform(shape=(32,),dtype=tf.float32))
            convd_2=tf.nn.conv2d(convd_pool_1,convd_w_2,strides=[1,1,1,1],padding="SAME")
            convd_2=tf.nn.relu(tf.add(convd_2,convd_b_2))
            convd_out=tf.nn.max_pool(convd_2,ksize=[1,2,1,1],strides=[1,2,1,1],padding="SAME")
            outs=tf.reshape(convd_out,[-1,100,32])
            outs=tf.transpose(outs,[0,2,1]) #[batch_size,32,100]
            encoder_state_c=tf.reduce_mean(outs,axis=1)
            encoder_state_h=tf.reduce_mean(outs,axis=1)
        elif FLAGS.encoder_mod=="lstmTF":
            encoder_inputs=tf.unstack(self.title,self.title_len,1)
            attention_states, encoder_state=tf.contrib.legacy_seq2seq.embedding_encoder(
                encoder_inputs=encoder_inputs,
                              cell=self.cell,
                              num_encoder_symbols=self.title_vocab_len,
                              embedding_size=self.init_dim)
            outs=attention_states
            encoder_state_c=encoder_state
            encoder_state_h=[]
        else:
            _logger.error("please input correct encoder_mod!!")

        return outs,encoder_state_c,encoder_state_h

    def Decoder(self,encoder_out,encoder_state_c,encoder_state_h):
        '''
        解码
        :param encoder_out: 
        :return: 
        '''
        def loop_function(prev, _):
            prev_symbol = tf.argmax(prev, 1)
            # Note that gradients will not propagate through the second parameter of
            # embedding_lookup.
            emb_prev = tf.nn.embedding_lookup(self.content_emb_decoder, prev_symbol)
            return emb_prev

        decoder_list = tf.unstack(self.content_emb_input, self.content_len, 1)
        encoder_state = (encoder_state_c, encoder_state_h)
        if FLAGS.encoder_mod in ['lstm','cnn']:

            if FLAGS.mod=='train':
                decoder_out, decoder_state = tf.contrib.legacy_seq2seq.attention_decoder(
                    decoder_inputs=decoder_list,
                    initial_state=encoder_state,
                    attention_states=encoder_out,
                    cell=self.cell,
                    output_size=None,
                )
            else:
                decoder_out, decoder_state = tf.contrib.legacy_seq2seq.attention_decoder(
                    decoder_inputs=decoder_list,
                    initial_state=encoder_state,
                    attention_states=encoder_out,
                    cell=self.cell,
                    output_size=None,
                    loop_function=loop_function
                )

        elif FLAGS.encoder_mod in ['bilstm']:

            decoder_list = tf.unstack(self.content_emb_input, self.content_len, 1)
            encoder_state = (encoder_state_c, encoder_state_h)
            decoder_out, decoder_state = tf.contrib.legacy_seq2seq.attention_decoder(
                decoder_inputs=decoder_list,
                initial_state=encoder_state,
                attention_states=encoder_out,
                cell=self.cell_decoder,
                output_size=None,
            )
        else:
            decoder_inputs=tf.unstack(self.content_input,self.content_len,1)
            decoder_out, decoder_state=tf.contrib.legacy_seq2seq.embedding_attention_decoder(
                decoder_inputs=decoder_inputs,
                initial_state=encoder_state_c,
                attention_states=encoder_out,
                cell=self.cell,
                num_symbols=self.content_vocab_len,
                embedding_size=self.init_dim,
                num_heads=1,
                output_size=None,
                output_projection=None,
                feed_previous=False,
                initial_state_attention=False)
        return decoder_out,decoder_state

    def Loss(self,logit_list,label):
        '''
        计算 损失
        :param logit: 
        :param label: 
        :return: 
        '''
        if FLAGS.encoder_mod=="bilstm":
            self.w = tf.Variable(tf.random_uniform(shape=(self.num_class, 2*self.hidden_dim), dtype=tf.float32))
            self.b = tf.Variable(tf.random_uniform(shape=(self.num_class,), dtype=tf.float32))
            logits=tf.stack(logit_list,1)
            ll = tf.einsum('ijk,kl->ijl', logits, tf.transpose(self.w))
            logit=tf.add(ll, self.b)
            self.softmax_logit = tf.nn.softmax(logit, 2)
            if not FLAGS.sample_loss:
                label_one_hot=tf.one_hot(label,self.content_vocab_len,1,0,2)
                loss=tf.losses.softmax_cross_entropy(logits=logit,onehot_labels=label_one_hot)
            else:
                labels = tf.unstack(label, self.content_len, 1)
                losses = []
                for logit, label in zip(logit_list, labels):
                    label = tf.reshape(label, (-1, 1))
                    loss = tf.nn.sampled_softmax_loss(weights=self.w,
                                                      biases=self.b,
                                                      labels=label,
                                                      inputs=logit,
                                                      num_sampled=FLAGS.num_samples,
                                                      num_classes=self.num_class)
                    losses.append(loss)
                losses = tf.stack(losses)
                loss = tf.reduce_mean(losses)
        else:
            self.w = tf.Variable(tf.random_uniform(shape=(self.num_class,  self.hidden_dim), dtype=tf.float32))
            self.b = tf.Variable(tf.random_uniform(shape=(self.num_class,), dtype=tf.float32))

            logits = tf.stack(logit_list, 1)
            ll = tf.einsum('ijk,kl->ijl', logits, tf.transpose(self.w))
            logit_ = tf.add(ll, self.b)
            self.softmax_logit=tf.nn.softmax(logit_,2)

            if not FLAGS.sample_loss:
                # label_one_hot = tf.one_hot(label, self.content_vocab_len, 1, 0, 2)
                # loss = tf.losses.softmax_cross_entropy(logits=logit_, onehot_labels=label_one_hot)
                mask=tf.sequence_mask(self.content_seq_vec,self.content_len)
                mod = tf.losses.Reduction().NONE
                slot_loss = tf.losses.sparse_softmax_cross_entropy(self.content_decoder, logit_, reduction=mod)
                losses = tf.boolean_mask(slot_loss, mask)
                loss = tf.reduce_mean(losses)

            else:
                labels = tf.unstack(label, self.content_len, 1)
                losses = []
                for logit, label in zip(logit_list, labels):
                    label = tf.reshape(label, (-1, 1))
                    loss = tf.nn.sampled_softmax_loss(weights=self.w,
                                                      biases=self.b,
                                                      labels=label,
                                                      inputs=logit,
                                                      num_sampled=FLAGS.num_samples,
                                                      num_classes=self.num_class)
                    losses.append(loss)
                losses = tf.stack(losses)
                loss = tf.reduce_mean(losses)
        return loss

    def __array_convert(self,beam_data,score,beam_path,beam_size,beam_flag="Med"):
        '''
        beam_data 矩阵转换函数 取最大的前5个概率
        :param beam_data: 
        :return: 
        '''
        beam_data=np.reshape(beam_data,(beam_data.shape[0],beam_data.shape[2]))
        allsocre=score
        # print("allsocre", allsocre)
        # print("beampath", beam_path)
        res=[]
        if beam_flag=="Beg":

            for index,ele in enumerate(beam_data):
                if index < beam_size:
                    eles=[[i,e] for i,e in enumerate(ele)]
                    eles.sort(key=lambda x:x[1],reverse=True)
                    res.append(eles[index][0])
                    path_=beam_path[index][:]
                    path_.append(eles[index][0])
                    beam_path[index]=path_
                    socre_=allsocre[index]
                    socre_+=eles[index][1]
                    allsocre[index]=socre_
            next_input=np.array([ e[-1] for e in beam_path])
            return beam_path,allsocre,next_input

        else:
            ss_score=[] # 5X5的分数
            ss_path=[] # 5X5的路径
            for index,ele in enumerate(beam_data):
                eles = [[i, e] for i, e in enumerate(ele)]
                eles.sort(key=lambda x: x[1], reverse=True)
                # eles=eles[:beam_size] #前一个解码后取beam_size个最大概率的解码
                eles = [eles[0]]
                socre_=[float(e[1])+allsocre[index] for e in eles]
                path_=[e[0] for e in eles]

                for e in path_:
                    ss=beam_path[index][:]
                    ss.append(e)
                    ss_path.append(ss)
                ss_score.extend(socre_)
            all_res=[[socre,path] for socre,path in zip(ss_score,ss_path)] # 全部的解码输出
            all_res.sort(key=lambda x:x[0],reverse=True)
            all_res=all_res[:beam_size] # 总共取beam_size个解码输出
            path=[e[1] for e in all_res]
            socre=[e[0] for e in all_res]
            next_input=np.array([ e[-1] for e in path])
            return path,socre,next_input

    def show_result(self,logit,label):
        for soft, content in zip(logit, label):
            pre = "".join(list(map(lambda x: self.id2content[int(x)], soft))).replace("NONE", "")
            content = "".join(list(map(lambda x: self.id2content[int(x)], content))).replace('NONE', "")
            print(pre, "---", content)
            print('\n')

    def train(self,dd):

        config = tf.ConfigProto(allow_soft_placement=True)
        saver = tf.train.Saver()

        with tf.Session(config=config) as sess:
            # saver.restore(sess,'%s'%FLAGS.model_dir)
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

                    decoder_mask = np.zeros_like(decoder_input)
                    for index, e in enumerate(decoder_len):
                        decoder_mask[index][:e + 1] = 1
                    losses, soft_logit, _, acc = sess.run([self.loss, self.soft_logit, self.opt, self.acc],
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

                    test_losses, test_soft_logit, test_acc = sess.run([self.loss, self.soft_logit, self.acc],
                                                                      feed_dict={
                                                                          self.encoder_input: test_encoder_input_batch,
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

    def decoder_acc(self,pre,label,rel_len):
        '''
        slot准确率
        :param pre:
        :param label:
        :return:
        '''
        pre=np.argmax(pre,2)
        ss=0.0
        for e,e1,rl in zip(pre,label,rel_len):
            num=0.0
            for i in range(rl):
                if e[i]==e1[i]:
                    num+=1.0
            acc=num/float(rl)
            ss+=acc
        return ss/float(len(pre))


    def beam_search_decoder_batch(self, dd):
        '''
        束搜索解码
        :return:
        '''
        self.beam_decoder()
        self.id2content = dd.id2content
        self.beam_size=FLAGS.beam_size
        config = tf.ConfigProto(device_count={"CPU": FLAGS.use_cpu_num},  # limit to num_cpu_core CPU usage
                                inter_op_parallelism_threads=8,
                                intra_op_parallelism_threads=8,
                                log_device_placement=False,
                                allow_soft_placement=True,
                                )

        saver = tf.train.Saver()
        self.mod = "beam_decoder"
        with tf.Session(config=config) as sess:
            saver.restore(sess, FLAGS.model_dir)

            id2content = dd.id2content
            # self.feed_previous = True
            for _ in range(1):
                return_content_input, return_content_decoder, return_title, return_content_len, return_title_len, loss_weight = dd.next_batch()


                for batch_index in range(return_title.shape[0]):
                    self.socre=[0.0]*self.beam_size
                    self.beam_path=[[1]]*self.beam_size

                    title_input=[return_title[batch_index]]*self.beam_size
                    title_seq_input=[return_title_len[batch_index]]*self.beam_size
                    title_input=np.array(title_input)
                    title_seq_input=np.array(title_seq_input)

                    init_encoder_out, init_encoder_state_c, init_encoder_state_h = sess.run(
                        [self.encoder_outs, self.encoder_state_c, self.encoder_state_h],
                        feed_dict={
                            self.title: title_input,
                            self.title_seq_vec: title_seq_input
                        })
                    init_beam_input = np.ones(shape=(self.beam_size,), dtype=np.int32)
                    beam_inputs = [init_beam_input]
                    beam_state_c = [init_encoder_state_c]
                    beam_state_h = [init_encoder_state_h]

                    for j in range(return_content_input.shape[1]):
                        state_, beam_softmax_ = sess.run([self.state, self.beam_softmax],
                                                         feed_dict={self.beam_state_c: beam_state_c[-1],
                                                                    self.beam_state_h: beam_state_h[-1],
                                                                    self.beam_inputs: beam_inputs[-1],
                                                                    self.beam_encoder: init_encoder_out})
                        if j==0:

                            path,socre,next_input=self.__array_convert(beam_softmax_,self.socre,self.beam_path,self.beam_size,beam_flag="Beg")
                            self.socre=socre
                            self.beam_path=path
                        else:
                            path, socre, next_input=self.__array_convert(beam_softmax_,self.socre,self.beam_path,self.beam_size,beam_flag="Med")
                            self.socre = socre
                            self.beam_path = path

                        beam_state_c.append(state_[0])
                        beam_state_h.append(state_[1])
                        beam_inputs.append(next_input)

                    print(self.socre)
                    print([id2content[e] for e in return_title[batch_index]])
                    print([list(map(lambda x:id2content[x],ele))for ele in self.beam_path])

                    # print(self.beam_path)
                    print([id2content[e] for e in return_content_decoder[batch_index]])
                    print('*'*10)

    def infer(self, dd,sent):
        '''
        束搜索解码
        :return:
        '''
        self.id2content = dd.id2content
        config = tf.ConfigProto(device_count={"CPU": FLAGS.use_cpu_num},  # limit to num_cpu_core CPU usage
                                inter_op_parallelism_threads=8,
                                intra_op_parallelism_threads=8,
                                log_device_placement=False,
                                allow_soft_placement=True,
                                )

        saver = tf.train.Saver()
        self.mod = "beam_decoder"
        with tf.Session(config=config) as sess:
            saver.restore(sess, FLAGS.model_dir)

            id2content = dd.id2content
            # self.feed_previous = True
            return_content_input, return_content_decoder, return_title, return_content_len, return_title_len, loss_weight = dd.get_sent(sent)
            soft_logit= sess.run(self.softmax_logit, feed_dict={
                self.content_input: return_content_input,
                self.content_decoder: return_content_decoder,
                self.title: return_title,
                self.content_seq_vec: return_content_len,
                self.title_seq_vec: return_title_len
            })

            pre = np.argmax(soft_logit, 2)
            print(pre)
            for ele in pre:
                s=''.join([id2content[e] for e in ele])
                print(s)
def main(_):
    # 本模型为目的是构建问答系统 因此问句为encoder 答案为decoder
    _logger.info("训练/验证/测试 数据预处理.....")
    dd = Data_deal.DataDealSeq(train_path='/train_sample.txt', test_path='/test.txt',
                               dev_path='/test.txt',
                               dim=FLAGS.embedding_dim,
                               batch_size=FLAGS.batch_size,
                               content_len=FLAGS.decoder_len,
                               title_len=FLAGS.encoder_len,
                               flag="train_new")


    content_vocab_size, title_vocab_size = dd.get_vocab_size()
    _logger.info("数据处理完毕")
    _logger.info("参数列表\n"
                 "train_dir:%s\nbatch_size:%s\nembedding_dim:%s\nEncoder_len:%s\n"
                 "Decoder_len:%s\nencoder_vocab_size:%s\ndecoder_vocab_size:%s\nhidden_dim%s "
                 % (FLAGS.train_dir, FLAGS.batch_size, FLAGS.embedding_dim, FLAGS.encoder_len,
                    FLAGS.decoder_len, title_vocab_size, content_vocab_size, FLAGS.hidden_dim))
    _logger.info('*' * 50 + "构建模型" + '*' * 50)
    model = Seq2Seq(hidden_dim=FLAGS.hidden_dim, init_dim=FLAGS.embedding_dim, content_len=FLAGS.decoder_len,
                    title_len=FLAGS.encoder_len,
                    con_vocab_len=content_vocab_size, ti_vocab_len=title_vocab_size, batch_size=FLAGS.batch_size)

    if FLAGS.mod=='train':
        model.train(dd)
    elif FLAGS.mod=='infer':
        _logger.info("预测")
        # model.beam_search_decoder_batch(dd)
        while True:
            sent=input('输入')
            model.infer(dd,sent)


if __name__ == '__main__':
    tf.app.run()




