import tensorflow as tf
import numpy as np
from data.Data_deal import DataDealSeq
from model import embedding,encoder_decoder,loss,encoder,decoder,decoder_1
from nltk import bleu
from numpy.random import randint
import random
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filemode='w')
_logger=logging.getLogger('seq2seq_rl')
class Config(object):
    '''
    默认配置
    '''
    learning_rate=0.01
    num_samples=5000
    batch_size=128
    encoder_len=10
    decoder_len=15
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
tf.app.flags.DEFINE_string("mod", "infer", "默认为训练") # true for prediction
tf.app.flags.DEFINE_boolean('use_MMI',config.use_MMI,"是否使用最大互信息来增加解码的多样性")
FLAGS = tf.app.flags.FLAGS



class Seq2SeqRl(object):

    def __init__(self,scope,encoder_word_num,decoder_word_num):

        with tf.variable_scope(name_or_scope=scope):
            self.encoder_word_num=encoder_word_num
            self.decoder_word_num=decoder_word_num
            self.__build_model__()



    def __build_model__(self):

        self.encoder_input=tf.placeholder(shape=(None,FLAGS.encoder_len),dtype=tf.int32,name='encoder_input')

        self.encoder_seq_len=tf.placeholder(shape=(None,),dtype=tf.int32,name='encoder_seq_len')

        self.decoder_input=tf.placeholder(shape=(None,FLAGS.decoder_len),dtype=tf.int32,name='decoder_input')

        self.decoder_mask=tf.placeholder(shape=(None,FLAGS.decoder_len),dtype=tf.int32,name='decoder_mask')

        self.decoder_seq_len=tf.placeholder(shape=(None,),dtype=tf.int32,name='decoder_seq_len')

        self.decoder_label=tf.placeholder(shape=(None,FLAGS.decoder_len),dtype=tf.int32,name='decoder_label')

        self.target_batch=tf.placeholder(shape=(None,FLAGS.decoder_len),dtype=tf.float32)
        self.action_batch=tf.placeholder(shape=(None,FLAGS.decoder_len),dtype=tf.int32)

        self.encoder_emb,_=embedding(self.encoder_input,self.encoder_word_num,FLAGS.embedding_dim,name='encoder_emb')
        self.decoder_emb,dec_emb=embedding(self.decoder_input,self.decoder_word_num,FLAGS.embedding_dim,name='decoder_emb')
        self.decoder_label_emb,_=embedding(self.decoder_label,self.decoder_word_num,FLAGS.embedding_dim,name='decoder_emb',reuse=True)

        self.encoder_cell = tf.contrib.rnn.LSTMCell(FLAGS.hidden_dim,
                                            initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123),
                                            state_is_tuple=True)
        #
        self.encoder_out,self.encoder_states,self.encoder_state_tuple=encoder(self.encoder_emb,FLAGS.encoder_len,self.encoder_seq_len,self.encoder_cell,name='lstm')

        self.outs=decoder_1(encoder_out=self.encoder_state_tuple,encoder_state=self.encoder_states,encoder_seq_len=self.encoder_seq_len,encoder_len=FLAGS.encoder_len,
                decoder_emb=self.decoder_emb,decoder_seq_len=self.decoder_seq_len,decoder_len=FLAGS.decoder_len,
                decoder_vocab_num=self.decoder_word_num,cell=self.encoder_cell,name='decoder',decoder_embedding=dec_emb,hidden_dim=FLAGS.hidden_dim)

        self.soft_logit = tf.nn.softmax(self.outs, 1)

        action_batch_one=tf.cast(tf.one_hot(self.action_batch,self.decoder_word_num,1,0,2),tf.float32)
        action_probs=tf.reduce_max(tf.multiply(self.soft_logit,action_batch_one),2)
        # print("action_probs",action_probs)


        self.loss=tf.reduce_mean(tf.squared_difference(action_probs,self.target_batch))


        # self.loss=loss(self.outs,self.decoder_label,self.decoder_word_num,sample_num=FLAGS.num_samples,decoder_seq_len=self.decoder_seq_len,
        #                decoder_len=FLAGS.decoder_len,decoder_mask=self.decoder_mask)
        self.opt=tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(self.loss)

        # self.outs=encoder_decoder(encoder_emb=self.encoder_input,decoder_emb=self.decoder_input,encoder_len=FLAGS.encoder_len,
        #                 decoder_len=FLAGS.decoder_len,cell=self.encoder_cell,hidden_dim=FLAGS.hidden_dim,
        #                 encoder_voab_num=self.encoder_word_num,decoer_vocab_num=self.decoder_word_num,feed_previous=True,name='encoder_decoder')
        #
        # self.soft_logit = tf.nn.softmax(self.outs, 1)
        #
        # self.loss=loss(self.outs,self.decoder_label,self.decoder_word_num,sample_num=FLAGS.num_samples,decoder_seq_len=self.decoder_seq_len,
        #                decoder_len=FLAGS.decoder_len,decoder_mask=self.decoder_mask)
        # self.opt=tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(self.loss)



    def predict(self,sess,state):
        '''
        根据状态得到action
        :param sess:
        :param state:
        :return:
        '''
        encoder_input=state['EnSen']['input']
        encoder_len=state['EnSen']['len']
        decoder_input=state['DeSen']['input']
        decoder_len=state['DeSen']['len']

        soft_logit=sess.run(self.soft_logit,feed_dict={self.encoder_input:encoder_input,
                                            self.encoder_seq_len:encoder_len,
                                            self.decoder_input:decoder_input,
                                            self.decoder_seq_len:decoder_len
                                            })

        return soft_logit[0]

    def predict_batch(self,sess,states):
        '''
        根据状态得到action
        :param sess:
        :param state:
        :return:
        '''
        encoder_inputs,encoder_lens,decoder_inputs,decoder_lens,steps=[],[],[],[],[]
        for state in states:
            encoder_input=state['EnSen']['input']
            encoder_len=state['EnSen']['len']
            decoder_input=state['DeSen']['input']
            decoder_len=state['DeSen']['len']

            encoder_inputs.extend(encoder_input)
            encoder_lens.extend(encoder_len)
            decoder_inputs.extend(decoder_input)
            decoder_lens.extend(decoder_len)

        encoder_inputs=np.array(encoder_inputs)
        encoder_lens=np.array(encoder_lens)
        decoder_inputs=np.array(decoder_inputs)
        decoder_lens=np.array(decoder_lens)


        soft_logit=sess.run(self.soft_logit,feed_dict={self.encoder_input:encoder_inputs,
                                            self.encoder_seq_len:encoder_lens,
                                            self.decoder_input:decoder_inputs,
                                            self.decoder_seq_len:decoder_lens
                                            })



        return soft_logit



    def infer(self,sess,encoder_input,encoder_len,decoder_input,decoder_len):

        soft_logit = sess.run(self.soft_logit, feed_dict={self.encoder_input: encoder_input,
                                                          self.encoder_seq_len: encoder_len,
                                                          self.decoder_input: decoder_input,
                                                          self.decoder_seq_len: decoder_len
                                                          })

        return np.argmax(soft_logit,2)


    def update(self,sess,state_batch,action_batch,target_batch):

        encoder_inputs,encoder_lens,decoder_inputs,decoder_lens,steps=[],[],[],[],[]
        for state in state_batch:
            encoder_input=state['EnSen']['input']
            encoder_len=state['EnSen']['len']
            decoder_input=state['DeSen']['input']
            decoder_len=state['DeSen']['len']

            encoder_inputs.extend(encoder_input)
            encoder_lens.extend(encoder_len)
            decoder_inputs.extend(decoder_input)
            decoder_lens.extend(decoder_len)

        encoder_inputs = np.array(encoder_inputs)
        encoder_lens = np.array(encoder_lens)
        decoder_inputs = np.array(decoder_inputs)
        decoder_lens = np.array(decoder_lens)
        steps = np.array(steps)


        loss,_=sess.run([self.loss,self.opt],feed_dict={
            self.encoder_input: encoder_inputs,
            self.encoder_seq_len: encoder_lens,
            self.decoder_input: decoder_inputs,
            self.decoder_seq_len: decoder_lens,
            self.action_batch:action_batch,
            self.target_batch:target_batch

        })
        return loss


def make_epsilon_greedy_policy(estimator, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.

    Args:
        estimator: An estimator that returns q values for a given state
        nA: Number of actions in the environment.

    Returns:
        A function that takes the (sess, observation, epsilon) as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """
    def policy_fn(sess, observation, epsilon,step):
        A = np.ones(nA, dtype=float) * epsilon / nA[1]
        s = random.uniform(0, 1)


        q_values = estimator.predict(sess=sess, state=observation)
        if s<=epsilon:
            print('随机选择action')
            best_action=randint(0,estimator.decoder_word_num,(1,FLAGS.decoder_len))[0]
            best_action_probs=q_values[1,best_action][0]
        else:
            best_action = np.argmax(q_values,1)
            best_action_probs=np.max(q_values,1)
        return best_action,best_action_probs
        # A[best_action] += (1.0 - epsilon)
        # return A
    return policy_fn


class ENV(object):

    def __init__(self):
        self.dd = DataDealSeq(train_path="/baidu_zd_500.txt", test_path="/test.txt",
                         dev_path="/test.txt",
                         dim=FLAGS.hidden_dim, batch_size=1, content_len=FLAGS.decoder_len,
                         title_len=FLAGS.encoder_len, flag="train_new")

        self.decoder_word_num = len(self.dd.content_vocab)
        self.encoder_word_num = len(self.dd.title_vocab)




    def get_init_state(self,sess,model):
        '''
        获得初始的state
        :return:
        '''
        decoder_input, decoder_label, encoder_input, decoder_len, encoder_len, _ = self.dd.next_batch()

        pred_decoder=model.infer(sess=sess,encoder_input=encoder_input,encoder_len=encoder_len,decoder_input=decoder_input,decoder_len=decoder_len)
        for i in range(1,decoder_len[0]):
            decoder_input[0][i]=pred_decoder[0][i-1]

        # new_decoder_input=np.zeros_like(pred_decoder)
        # for i in range(new_decoder_input.shape[1]):
        #     if i==0:
        #         new_decoder_input[0][i]=decoder_input[0][i]
        #     else:
        #         new_decoder_input[0][i]=pred_decoder[0][i]


        return {'EnSen':{'input':encoder_input,'len':encoder_len},'DeSen':{'input':decoder_input,'len':decoder_len},'TrSen':decoder_label,'step':0}


    def compute_bleu(self,true_dec,this_dec,decoder_len):
        # this_reward = bleu([true_dec], this_dec)
        # num=sum([1 for e in this_dec[:decoder_len] if e in true_dec[:decoder_len]])
        tt=true_dec[:decoder_len]
        pp=this_dec[:decoder_len]
        num=sum([1 for e,e1 in zip(tt,pp) if e==e1])
        this_reward=float(num)/float(len(true_dec[:decoder_len]))
        return this_reward

    def step(self,action,state,step,max_turn):
        '''
        根据反馈的action获取下一个state
        :param action:
        :return:
        '''
        done=False
        state['step']=step
        decoder_len=state['DeSen']['len'][0]
        true_dec=list(state['TrSen'][0])
        this_dec = list(state['DeSen']['input'][0])
        max_reward=0
        max_index=0
        max_ele=0
        for index,ele  in enumerate(action):
            this_dec[index]=ele

            this_reward=self.compute_bleu(true_dec,this_dec,decoder_len)

            if this_reward>max_reward:
                max_reward=this_reward
                max_index=index
                max_ele=ele


        state['DeSen']['input'][0][max_index]=max_ele


        # true_dec=[str(e) for e in true_dec]
        # this_dec=[str(e) for e in this_dec]

        print('true_dec',true_dec)
        print('this_dec',this_dec)



        # bleu_score=bleu([true_dec],this_dec)
        # if bleu_score==0:
        #     reward=-2
        # elif bleu_score<0.8:
        #     reward=-1
        # else:
        #     reward = bleu_score

        # reward=sum(1 for e in np.equal(np.array(true_dec),np.array(this_dec)) if e)/len(true_dec)
        # reward=bleu_score

        if  step==max_turn:
            done = True


        return state,max_reward,done,{max_index:max_ele},action

    def get_q_value_batch(self,state_batch,action_probs_batch):

        reward_batch=[]
        action_batch=[]
        for state,action in zip(state_batch,action_probs_batch):
            true_dec = list(state['TrSen'][0])
            this_dec = list(state['DeSen']['input'][0])
            max_reward = 0
            max_index = 0
            max_ele = 0
            max_probs=0.0
            for index, ele in enumerate(action):
                this_dec[index] = ele

                this_reward = bleu([true_dec], this_dec)

                if this_reward > max_reward:
                    max_reward = this_reward
                    max_index = index
                    max_ele = ele

            reward_batch.append(max_reward)
            action_batch.append({max_index:max_ele})
        return reward_batch,action_batch

    def batch_step(self,batch_action,batch_state,turn,max_turn):

        batch_next_state=[]
        batch_reward=[]
        batch_done=[]
        for action,state in zip(batch_action,batch_state):
            next_state,reward,done=self.step(action,state,turn,max_turn)
            batch_next_state.append(next_state)
            batch_reward.append(reward)
            batch_done.append(done)

        batch_next_state=np.array(batch_next_state)
        batch_reward=np.array(batch_reward)
        batch_done=np.array(batch_done)
        return batch_next_state,batch_reward,batch_done

    def get_train_state(self):
        '''
        获取 train集的 state
        :return:
        '''
        train_sent, train_slot, train_intent, train_rel_len, train_index = self.dd.get_train()
        train_sent_action = np.ones_like(train_rel_len)
        train_sent_action = train_sent_action * self.intent_num
        res=[]
        for i in range(train_sent.shape[0]):
            res.append({'sent_array':train_sent[i],'sent_vec':train_rel_len[i],'label':train_intent[i],'sent_action':train_sent_action[i]})
        return res

    def get_dev_state(self):
        '''
        获取dev的init_state
        :return:
        '''
        dev_sent, dev_slot, dev_intent, dev_rel_len, dev_index = self.dd.get_dev()
        dev_sent_action = np.ones_like(dev_rel_len)
        dev_sent_action = dev_sent_action * self.intent_num

        return {'sent_array': dev_sent, 'sent_vec': dev_rel_len, 'label': dev_intent,
                'sent_action': dev_sent_action}


def copy_model_parameters(sess, estimator1, estimator2):
    """
    Copies the model parameters of one estimator to another.

    Args:
      sess: Tensorflow session instance
      estimator1: Estimator to copy the paramters from
      estimator2: Estimator to copy the parameters to
    """
    e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]
    e1_params = sorted(e1_params, key=lambda v: v.name)
    e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]
    e2_params = sorted(e2_params, key=lambda v: v.name)

    update_ops = []
    for e1_v, e2_v in zip(e1_params, e2_params):
        op = e2_v.assign(e1_v)
        update_ops.append(op)

    sess.run(update_ops)



def deep_q_learning(sess,
                    env,
                    q_estimator,
                    target_estimator,
                    num_episodes,
                    experiment_dir,
                    replay_memory_size=50,
                    replay_memory_init_size=50,
                    update_target_estimator_every=50,
                    discount_factor=0.2,
                    epsilon_start=0.3,
                    epsilon_end=0.1,
                    epsilon_decay_steps=500,
                    batch_size=32,
                    record_video_every=50,
                    max_turn=FLAGS.decoder_len):
    """
    Q-Learning algorithm for off-policy TD control using Function Approximation.
    Finds the optimal greedy policy while following an epsilon-greedy policy.

    Args:
        sess: Tensorflow Session object
        env: OpenAI environment
        q_estimator: Estimator object used for the q values
        target_estimator: Estimator object used for the targets
        state_processor: A StateProcessor object
        num_episodes: Number of episodes to run for
        experiment_dir: Directory to save Tensorflow summaries in
        replay_memory_size: Size of the replay memory
        replay_memory_init_size: Number of random experiences to sampel when initializing
          the reply memory.
        update_target_estimator_every: Copy parameters from the Q estimator to the
          target estimator every N steps
        discount_factor: Gamma discount factor
        epsilon_start: Chance to sample a random action when taking an action.
          Epsilon is decayed over time and this is the start value
        epsilon_end: The final minimum value of epsilon after decaying is done
        epsilon_decay_steps: Number of steps to decay epsilon over
        batch_size: Size of batches to sample from the replay memory
        record_video_every: Record a video every N episodes

    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    print('开始DQN')
    #有效的action 列表
    VALID_ACTIONS=list(range(q_estimator.decoder_word_num+1))
    stats={}

    # The replay memory
    # 经验池
    replay_memory = []

    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)
    print('epsion',epsilons.shape)
    # The policy we're following
    policy = make_epsilon_greedy_policy(
        q_estimator,
        (FLAGS.decoder_len,q_estimator.decoder_word_num))

    # Populate the replay memory with initial experience
    print("填充 经验池 ...")
    state = env.get_init_state(sess=sess,model=q_estimator) #获取初始state:{sent_arry,sent_vec,label,sent_action}
    print(state)


    #选择性加载权重
    # var = tf.global_variables()
    # var_to_restore = [val for val in var if 'seqRl' in val.name]
    # save = tf.train.Saver(var_to_restore)
    # ckpt = tf.train.get_checkpoint_state('./save_model/')
    #
    # save.restore(sess, './save_model/seq2seq_bilstm_.ckpt.data-00000-of-00001')#载入数据
    save = tf.train.Saver()
    save.restore(sess,'./save_model/seq2seq_bilstm_.ckpt')
    #
    # save.restore(sess,'./rl_save_model/seq2seq_rl.ckpt')


    for i in range(replay_memory_init_size):
        turn=0
        for step in range(1,2*FLAGS.decoder_len):
            actions,action_probs = policy(sess=sess, observation=state, epsilon=epsilons[min(0, epsilon_decay_steps-1)],step=step)
            print('action:{}'.format(actions))
            # action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done,action_dict,action= env.step(actions,state,step,max_turn)
            print('reward{}'.format(reward))
            turn += 1
            replay_memory.append((state, action, reward, next_state, done))
            if done or turn>max_turn:
                state = env.get_init_state(sess=sess, model=q_estimator)
                turn=0
            else:
                state = next_state


    total_t = 0
    init_loss=999
    for i_episode in range(200):

        # Reset the environment
        state = env.get_init_state(sess=sess, model=q_estimator)


        # One step in the environment
        turn=0
        step=1
        for t in range(1,2*FLAGS.decoder_len):
            _logger.info('\n\n')
            _logger.info('第%s次 第%s迭代'%(i_episode,t))
            # Epsilon for this time step
            epsilon = epsilons[min(total_t, epsilon_decay_steps-1)]

            # if total_t % update_target_estimator_every == 0:
            #     copy_model_parameters(sess, q_estimator, target_estimator)
            #     print("\nCopied model parameters to target network.")

            # Take a step
            actions,action_probs = policy(sess, state, epsilon,step)
            # action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

            # print('\n','###### predict',,action)
            next_state, reward, done,action_dict,action = env.step(actions,state,step,max_turn)
            print(next_state)
            print('new reward{}'.format(reward))
            turn+=1
            step += 1
            if turn>=max_turn or step>=FLAGS.decoder_len:
                turn=0
                step=1
                state = env.get_init_state(sess=sess, model=q_estimator)

            # If our replay memory is full, pop the first element
            if len(replay_memory) == replay_memory_size:
                replay_memory.pop(0)

            # Save transition to replay memory
            replay_memory.append((state, action, reward, next_state, done))

            # Update statistics
            if i_episode in stats:
                w=stats[i_episode]['episode_rewards']
                w+=reward
                stats[i_episode]['episode_rewards']=w
                stats[i_episode]['episode_lengths'] = t
            else:
                stats[i_episode]={"episode_rewards":reward,"episode_lengths":t}

            # Sample a minibatch from the replay memory
            for _ in range(3):
                np.random.shuffle(replay_memory)
                samples=replay_memory[:batch_size]
                # samples = random.sample(replay_memory, batch_size)
                states_batch, action_batch, reward_batch, next_states_batch, done_batch=[],[],[],[],[]
                for ele in samples:
                    states_batch.append(ele[0])
                    action_batch.append(ele[1])
                    reward_batch.append(ele[2])
                    next_states_batch.append(ele[3])
                    done_batch.append(ele[4])
    #
                # states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))

                # Calculate q values and targets (Double DQN)
                # q_values_next = q_estimator.predict_batch(sess, next_states_batch)
                # best_actions_probs = np.max(q_values_next, axis=2)
                # targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * \
                #     discount_factor * best_actions_probs

                action_prob_batch = q_estimator.predict_batch(sess, next_states_batch)
                q_values_next=np.max(action_prob_batch,2)
                print('q_values_next',q_values_next.shape)

                targets_batch=np.reshape(reward_batch,(batch_size,1))+discount_factor*np.array(q_values_next)
                # q_values_next_target = q_estimator.predict_batch(sess, next_states_batch) #[batch,seq_len,word_num]
                # # targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * \
                # #                 discount_factor * q_values_next_target[np.arange(batch_size), best_actions]
                # reward_batch=np.reshape(reward_batch,(batch_size,1))
                # targets_batch=reward_batch+discount_factor*np.max(q_values_next,2)


                #             # Perform gradient descent update
                states_batch = np.array(states_batch)
                loss = q_estimator.update(sess, states_batch, action_batch, targets_batch)
                if loss<init_loss:
                    init_loss=loss
                    save.save(sess,'./rl_save_model/seq2seq_rl.ckpt')
                    print('save model')

                print('loss{}'.format(loss))
    #             print("\rStep {} ({}) @ Episode {}/{}, loss: {}".format(
    #                 t, total_t, i_episode + 1, num_episodes, loss), end="")
    #             print('targets_batch:{}'.format(targets_batch),'\n')
    #             num=float(sum([1 for e in reward_batch if e>=1.0]))/float(len(reward_batch))
    #             print('success:{}'.format(num))
    #             print('done:{}'.format(done_batch))
    #             if done:
    #                 break
    #
    #         state = next_state
    #         total_t += 1
    #
    #     turn=0
    #     train_state=env.get_train_state()
    #     dev_state=env.get_dev_state()
    #     for i in range(max_turn):
    #         train_action_probs = policy(sess=sess, observation=train_state, epsilon=epsilons[min(0, epsilon_decay_steps - 1)])
    #         train_next_state, train_reward, train_done = env.step(train_action_probs, train_state, i, max_turn)
    #
    #         dev_action_probs = policy(sess=sess, observation=dev_state,
    #                                    epsilon=epsilons[min(0, epsilon_decay_steps - 1)])
    #         dev_next_state, dev_reward, dev_done = env.step(dev_action_probs, dev_state, i, max_turn)
    #
    #         train_state=train_next_state
    #         dev_state=dev_next_state
    #
    #
    #     train_logit=q_estimator.predict(sess=sess,state=train_state)
    #     dev_logit=q_estimator.predict(sess=sess,state=dev_state)
    #
    #     print('train_logit:{}'.format(train_logit))
    #
    #
    #
    #     # dev_acc,train_acc=q_estimator.dev_predict(sess)
    #     # print('#'*10,'dev_acc:{}  train:{}'.format(dev_acc,train_acc))
    #
    # #     # Add summaries to tensorboard
    # #     episode_summary = tf.Summary()
    # #     episode_summary.value.add(simple_value=stats.episode_rewards[i_episode], node_name="episode_reward", tag="episode_reward")
    # #     episode_summary.value.add(simple_value=stats.episode_lengths[i_episode], node_name="episode_length", tag="episode_length")
    # #     q_estimator.summary_writer.add_summary(episode_summary, total_t)
    # #     q_estimator.summary_writer.flush()
    # #
    # #     yield total_t, plotting.EpisodeStats(
    # #         episode_lengths=stats.episode_lengths[:i_episode+1],
    # #         episode_rewards=stats.episode_rewards[:i_episode+1])
    # #
    # # env.monitor.close()
    # # return stats
    #

def main(_):
    tf.reset_default_graph()
    config = tf.ConfigProto(allow_soft_placement=True)
    env = ENV()
    q_estimator = Seq2SeqRl(scope='seqRl', encoder_word_num=env.encoder_word_num, decoder_word_num=env.decoder_word_num)
    # target_estimator = Seq2SeqRl(scope='seqRl_target', encoder_word_num=env.encoder_word_num,
    #                              decoder_word_num=env.decoder_word_num)
    target_estimator=None
    saver=tf.train.Saver()
    with tf.Session(config=config) as sess:

        sess.run(tf.global_variables_initializer())
        # saver.restore(sess,'./save_model/seq2seq_bilstm_.ckpt')
        deep_q_learning(sess=sess,
                        env=env,
                        q_estimator=q_estimator,
                        target_estimator=target_estimator,
                        num_episodes=100,
                        experiment_dir=None)


if __name__ == '__main__':
    tf.app.run()
