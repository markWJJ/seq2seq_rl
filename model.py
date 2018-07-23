

import tensorflow as tf


def embedding(input,vocab_num,dim,name,reuse=False):

    with tf.variable_scope(name_or_scope=name,reuse=reuse):

        emb=tf.get_variable(name=name,shape=(vocab_num,dim),initializer=tf.random_normal_initializer())

        out=tf.nn.embedding_lookup(emb,input)
        return out,emb


def encoder_decoder(encoder_emb,encoder_len,decoder_emb,decoder_len,cell,hidden_dim,encoder_voab_num,decoer_vocab_num,name,feed_previous):

    with tf.variable_scope(name_or_scope=name):
        encoder_inputs = tf.unstack(encoder_emb, encoder_len, 1)
        decoder_inputs = tf.unstack(decoder_emb, decoder_len, 1)
        out_w = tf.Variable(tf.random_uniform(shape=(hidden_dim, decoer_vocab_num), maxval=1.0, minval=-1.0),
                            dtype=tf.float32)
        out_b = tf.Variable(tf.random_uniform(shape=(decoer_vocab_num,)), dtype=tf.float32)
        outs, state = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
            encoder_inputs,
            decoder_inputs,
            cell,
            num_encoder_symbols=encoder_voab_num,
            num_decoder_symbols=decoer_vocab_num,
            embedding_size=hidden_dim,
            output_projection=(out_w, out_b),
            feed_previous=feed_previous,
            dtype=tf.float32)
        # self.loss=self.Loss(outs,self.content_decoder)
        # self.opt=tf.train.AdamOptimizer(0.03).minimize(self.loss)
        outs=tf.stack(outs,1)
        outs = tf.layers.dense(outs, decoer_vocab_num, use_bias=True)
        return outs


def encoder(encoder_emb,encoder_len,encoder_seq_len,cell,name='lstm'):

    with tf.variable_scope(name_or_scope=name):

        if name =='lstm':


            encoder_inputs=tf.unstack(encoder_emb,encoder_len,1)
            out,state=tf.contrib.rnn.static_rnn(cell,encoder_inputs,dtype=tf.float32,sequence_length=encoder_seq_len)
            out=tf.stack(out,1) #[b,s,emb]

            encoder_seq_len_hot=tf.cast(tf.expand_dims(tf.one_hot(encoder_seq_len,encoder_len,1,0,1),-1),tf.float32)   #[b,s,1]

            s=tf.multiply(out,encoder_seq_len_hot)

            s_mean=tf.reduce_mean(s,1)
            return s_mean,s,state


def __attention(input_t,attention_arry,name):

    with tf.variable_scope(name_or_scope=name,reuse=True):

        w_t=tf.get_variable(name='w_t',shape=(input_t.get_shape()[-1],100),dtype=tf.float32,initializer=tf.random_normal_initializer())
        w_a=tf.get_variable(name='w_a',shape=(input_t.get_shape()[-1],100),dtype=tf.float32,initializer=tf.random_normal_initializer())


        tf.einsum('')

def decoder(encoder_out,encoder_state,encoder_seq_len,encoder_len,decoder_emb,decoder_seq_len,decoder_len,
            name,decoder_vocab_num,cell,decoder_embedding,hidden_dim):


    with tf.variable_scope(name_or_scope=name):
        decoder_cell = tf.contrib.rnn.LSTMCell(decoder_vocab_num,state_is_tuple=True)
        def loop_function(prev, _):
            prev_symbol = tf.argmax(prev, 1)
            # Note that gradients will not propagate through the second parameter of
            # embedding_lookup.
            emb_prev = tf.nn.embedding_lookup(decoder_embedding, prev_symbol)
            return emb_prev
        decoder_list=tf.unstack(decoder_emb,decoder_len,1)
        decoder_out, decoder_state = tf.contrib.legacy_seq2seq.attention_decoder(
            decoder_inputs=decoder_list,
            initial_state=encoder_out,
            attention_states=encoder_state,
            cell=cell,
            output_size=decoder_vocab_num,
            loop_function=loop_function,

        )
        outs = tf.stack(decoder_out, 1)
        return outs
    # with tf.variable_scope(name_or_scope=name):
    #
    #     init_h=encoder_out
    #     init_c=encoder_out
    #
    #     H=[init_h]
    #     C=[init_c]
    #
    #     decoder_embs=tf.unstack(decoder_emb,decoder_len,1)
    #     decoder_cell = tf.contrib.rnn.LSTMCell(tf.cast(init_h, tf.float32).get_shape()[-1])
    #
    #     with tf.variable_scope(name_or_scope=name) as scope:
    #         for i in range(decoder_len):
    #             if i>0:
    #                 scope.reuse_variables()


def decoder_1(encoder_out,encoder_state,encoder_seq_len,encoder_len,decoder_emb,decoder_seq_len,decoder_len,
            name,decoder_vocab_num,cell,decoder_embedding,hidden_dim):


    with tf.variable_scope(name_or_scope=name):
        decoder_cell = tf.contrib.rnn.LSTMCell(decoder_vocab_num,state_is_tuple=True)
        def loop_function(prev, _):
            prev_symbol = tf.argmax(prev, 1)
            # Note that gradients will not propagate through the second parameter of
            # embedding_lookup.
            emb_prev = tf.nn.embedding_lookup(decoder_embedding, prev_symbol)
            return emb_prev
        decoder_list=tf.unstack(decoder_emb,decoder_len,1)
        decoder_out, decoder_state = tf.contrib.legacy_seq2seq.attention_decoder(
            decoder_inputs=decoder_list,
            initial_state=encoder_out,
            attention_states=encoder_state,
            cell=cell,
            output_size=decoder_vocab_num,
            loop_function=None,

        )
        outs = tf.stack(decoder_out, 1)
        return outs

def loss(logit,true_label,decoder_vocab_num,sample_num,decoder_seq_len,decoder_len,decoder_mask):

    label=tf.one_hot(true_label,decoder_vocab_num,1,0,axis=2)

    loss=tf.losses.softmax_cross_entropy(onehot_labels=label,logits=logit,reduction=tf.losses.Reduction.NONE)
    loss=tf.reduce_mean(tf.multiply(loss,tf.cast(decoder_mask,tf.float32)))
    #
    # loss=tf.losses.softmax_cross_entropy(onehot_labels=label,logits=logit)


    return loss