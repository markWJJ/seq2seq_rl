

import tensorflow as tf


def embedding(input,vocab_num,dim,name,reuse=False):

    with tf.variable_scope(name_or_scope=name,reuse=reuse):

        emb=tf.get_variable(name=name,shape=(vocab_num,dim),initializer=tf.random_normal_initializer(),trainable=True)

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


def last_relevant_output(output, sequence_length):
    """
    Given the outputs of a LSTM, get the last relevant output that
    is not padding. We assume that the last 2 dimensions of the input
    represent (sequence_length, hidden_size).

    Parameters
    ----------
    output: Tensor
        A tensor, generally the output of a tensorflow RNN.
        The tensor index sequence_lengths+1 is selected for each
        instance in the output.

    sequence_length: Tensor
        A tensor of dimension (batch_size, ) indicating the length
        of the sequences before padding was applied.

    Returns
    -------
    last_relevant_output: Tensor
        The last relevant output (last element of the sequence), as retrieved
        by the output Tensor and indicated by the sequence_length Tensor.
    """
    with tf.name_scope("last_relevant_output"):
        batch_size = tf.shape(output)[0]
        max_length = tf.shape(output)[-2] #seq_len
        out_size = int(output.get_shape()[-1]) #dim
        index = tf.range(0, batch_size) * max_length + (sequence_length - 1)
        print(index)
        flat = tf.reshape(output, [-1, out_size])
        relevant = tf.gather(flat, index)
        return relevant


def mean_pool(input_tensor, sequence_length=None):
    """
    Given an input tensor (e.g., the outputs of a LSTM), do mean pooling
    over the last dimension of the input.

    For example, if the input was the output of a LSTM of shape
    (batch_size, sequence length, hidden_dim), this would
    calculate a mean pooling over the last dimension (taking the padding
    into account, if provided) to output a tensor of shape
    (batch_size, hidden_dim).

    Parameters
    ----------
    input_tensor: Tensor
        An input tensor, preferably the output of a tensorflow RNN.
        The mean-pooled representation of this output will be calculated
        over the last dimension.

    sequence_length: Tensor, optional (default=None)
        A tensor of dimension (batch_size, ) indicating the length
        of the sequences before padding was applied.

    Returns
    -------
    mean_pooled_output: Tensor
        A tensor of one less dimension than the input, with the size of the
        last dimension equal to the hidden dimension state size.
    """
    with tf.name_scope("mean_pool"):
        # shape (batch_size, sequence_length)
        input_tensor_sum = tf.reduce_sum(input_tensor, axis=-2)

        # If sequence_length is None, divide by the sequence length
        # as indicated by the input tensor.
        if sequence_length is None:
            sequence_length = tf.shape(input_tensor)[-2]

        # Expand sequence length from shape (batch_size,) to
        # (batch_size, 1) for broadcasting to work.
        expanded_sequence_length = tf.cast(tf.expand_dims(sequence_length, -1),
                                           "float32") + 1e-08

        # Now, divide by the length of each sequence.
        # shape (batch_size, sequence_length)
        mean_pooled_input = (input_tensor_sum /
                             expanded_sequence_length)
        return mean_pooled_input

def decoder_1(encoder_out,encoder_state,encoder_seq_len,encoder_len,decoder_emb,decoder_seq_len,decoder_len,
            name,decoder_vocab_num,cell,decoder_embedding,hidden_dim,mod):


    with tf.variable_scope(name_or_scope=name):
        decoder_cell = tf.contrib.rnn.LSTMCell(decoder_vocab_num,state_is_tuple=True)
        def loop_function(prev, _):
            prev_symbol = tf.argmax(prev, 1)
            # Note that gradients will not propagate through the second parameter of
            # embedding_lookup.
            emb_prev = tf.nn.embedding_lookup(decoder_embedding, prev_symbol)
            return emb_prev
        decoder_list=tf.unstack(decoder_emb,decoder_len,1)
        loop_function1 = loop_function
        if mod=='infer':
            loop_function1=loop_function


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

def loss(logit,true_label,decoder_vocab_num,sample_num,decoder_seq_len,decoder_len,decoder_mask):

    label=tf.one_hot(true_label,decoder_vocab_num,1,0,axis=2)
    
    loss=tf.losses.softmax_cross_entropy(onehot_labels=label,logits=logit,reduction=tf.losses.Reduction.NONE)
    loss=tf.reduce_mean(tf.multiply(loss,tf.cast(decoder_mask,tf.float32)))
    #
    # loss=tf.losses.softmax_cross_entropy(onehot_labels=label,logits=logit)


    return loss