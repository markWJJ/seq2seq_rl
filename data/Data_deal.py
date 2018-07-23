import numpy as np
import pickle
import os
global PATH
import sys
import jieba
from sklearn.cross_validation import _shuffle
PATH=os.path.split(os.path.realpath(__file__))[0]


class DataDealSeq(object):
    def __init__(self,train_path,dev_path,test_path,dim,batch_size,content_len,title_len,flag):
        self.train_path=train_path
        self.dev_path=dev_path
        self.test_path=test_path
        self.dim=dim
        self.content_len=content_len
        self.title_len=title_len
        self.batch_size=batch_size
        self.num_batch=1

        if flag=="train_new":
            self.content_vocab,self.title_vocab=self.get_vocab()
            pickle.dump(self.content_vocab,open(PATH+"/content_vocab.p",'wb'))
            pickle.dump(self.title_vocab,open(PATH+"/title_vocab.p",'wb'))
        elif flag=="test" or flag=="train":
            self.content_vocab=pickle.load(open(PATH+"/content_vocab.p",'rb'))
            self.title_vocab=pickle.load(open(PATH+"/title_vocab.p",'rb'))

        self.id2content={}
        for k,v in self.content_vocab.items():
            self.id2content[v]=k
        self.id2encoder={}
        for k,v in self.title_vocab.items():
            self.id2encoder[v]=k
        self.index=0

    def get_vocab(self):
        '''
        构造字典
        :return: 
        '''
        train_file=open(PATH+self.train_path,'r')
        test_file=open(PATH+self.dev_path,'r')
        dev_file=open(PATH+self.test_path,'r')
        content_vocab={"NONE":0,"BEG":1,"END":2}
        title_vocab={"NONE":0}
        content_index=3
        title_index=1
        ss=0
        for ele in train_file:
            ss+=1
            ele=ele.replace("\n","")
            try:
                content=ele.split("\t\t")[1]
                title=ele.split("\t\t")[0]
            except:
                print(ele)
            for t in title.split(" "):
                if t and t not in title_vocab:
                    title_vocab[t]=title_index
                    title_index+=1

            for c in content.split(" "):
                if c and c not in content_vocab:
                    content_vocab[c]=content_index
                    content_index+=1

        for ele in dev_file:
            ss+=1
            ele.replace("\n","")
            content=ele.split("\t\t")[1]
            title=ele.split("\t\t")[0]
            for t in title.split(" "):
                if t and t not in title_vocab:
                    title_vocab[t]=title_index
                    title_index+=1

            for c in content.split(" "):
                if c and c not in content_vocab:
                    content_vocab[c]=content_index
                    content_index+=1

        train_file.close()
        dev_file.close()
        self.num_batch=int(float(ss)/float(self.batch_size))
        return content_vocab,title_vocab

    def sent2vec_content(self,sent,max_len):
        '''
        根据vocab将content转换为向量，并标注开始/结束标签
        :param sent: 
        :return: 
        '''
        sent=str(sent).replace("\n","")
        real_len=len(sent.split(" "))
        vocab=self.content_vocab
        sent_list=[]
        for word in sent.split(" "):
            word=word.replace(" ","")
            if word:
                if word in vocab:
                    sent_list.append(vocab[word])
                else:
                    sent_list.append(vocab["NONE"])

        if len(sent_list)>=max_len:
            new_sent_list=sent_list[:max_len]
            ss_in=[vocab['BEG']]
            ss_in.extend(new_sent_list)
            sent_list_input=ss_in[:-1]

            ss_de=sent_list[:max_len-1]
            ss_de.extend([vocab['END']])
            sent_list_decoder=ss_de[:max_len]

        else:

            beg=[vocab['BEG']]
            beg.extend(sent_list)
            sent_list_input=beg

            end=sent_list[:]
            end.extend([vocab['END']])

            sent_list_decoder=end

            ss=[0]*(max_len-len(sent_list_input))
            sent_list_input.extend(ss)
            sent_list_decoder.extend(ss)
            sent_list_input=sent_list_input[:max_len]
            sent_list_decoder=sent_list_decoder[:max_len]


        sent_vec_input=np.array(sent_list_input)
        sent_vec_decoder=np.array(sent_list_decoder)
        return sent_vec_input,sent_vec_decoder,real_len

    def sent2vec_title(self,sent,max_len):
        '''
        根据vocab将句子转换为向量
        :param sent: 
        :return: 
        '''
        sent=str(sent).replace("\n","")
        sent_list=[]
        real_len=len(sent.split(" "))
        vocab={}
        vocab=self.title_vocab
        for word in sent.split(" "):
            word=word.replace(" ","")
            if word and word in vocab:
                sent_list.append(vocab[word])
            else:
                sent_list.append(0)
        if len(sent_list)>=max_len:
            new_sent_list=sent_list[0:max_len]
        else:
            new_sent_list=sent_list
            ss=[0]*(max_len-len(sent_list))
            new_sent_list.extend(ss)
        sent_vec=np.array(new_sent_list)
        return sent_vec,real_len

    def shuffle(self,*args):
        '''
        将矩阵X打乱
        :param x: 
        :return: 
        '''
        ss=list(range(args[0].shape[0]))
        np.random.shuffle(ss)
        new_res=[]
        for e in args:
            new_res.append(np.zeros_like(e))
        fin_res=[]
        for index,ele in enumerate(new_res):
            for i in range(args[0].shape[0]):
                ele[i]=args[index][ss[i]]
            fin_res.append(ele)
        return fin_res


    def shuffle1(self,Q,Q_,A):
        '''
        将矩阵X打乱
        :param x: 
        :return: 
        '''
        ss=list(range(Q.shape[0]))
        np.random.shuffle(ss)
        new_Q=np.zeros_like(Q)
        new_A=np.zeros_like(A)
        new_Q_=np.zeros_like(Q_)
        for i in range(Q.shape[0]):
            new_Q[i]=Q[ss[i]]
            new_A[i]=A[ss[i]]
            new_Q_[i]=Q_[ss[i]]

        return new_Q,new_Q_,new_A

    def get_ev_ans(self,sentence):
        '''
        获取 envience and answer_label
        :param sentence: 
        :return: 
        '''
        env_list=[]
        ans_list=[]
        for e in sentence.split(" "):
            try:
                env_list.append(e.split("/")[0])
                ans_list.append(self.label_dict[str(e.split("/")[1])])
            except:
                pass
        return " ".join(env_list),ans_list

    def next_batch(self):
        '''
        获取训练机的下一个batch
        :return: 
        '''

        train_file=open(PATH+self.train_path,'r')

        content_input_list=[]
        content_decoder_list=[]
        title_list=[]
        content_len_list=[]
        title_len_list=[]
        loss_weights=[]

        train_sentcens=train_file.readlines()
        file_size=len(train_sentcens)

        for sentence in train_sentcens:
            sentence = sentence.replace("\n", "")
            sentences = sentence.split("\t\t")
            try:
                content_sentence = sentences[1]
                title_sentence=sentences[0]
            except:
                print(sentence)

            content_array_input,content_array_decoder,_=self.sent2vec_content(content_sentence,self.content_len)
            title_array,_=self.sent2vec_title(title_sentence,self.title_len)

            loss_weight=np.zeros_like(content_array_decoder,dtype=np.float32)
            content_len=len(content_sentence.split(" "))
            loss_weight[:content_len]=1.0
            loss_weights.append(loss_weight)
            content_input_list.append(list(content_array_input))
            content_decoder_list.append(list(content_array_decoder))
            title_list.append(list(title_array))
            if len(str(content_sentence).split(" "))>=self.content_len:
                content_len_list.append(self.content_len)
            else:
                content_len_list.append(len(str(content_sentence).split(" ")))

            if len(str(title_sentence).split(" ")) >= self.title_len:
                title_len_list.append(self.title_len)
            else:
                title_len_list.append(len(str(title_sentence).split(" ")))


        train_file.close()
        result_content_input=np.array(content_input_list,dtype=np.int32)
        result_content_decoder=np.array(content_decoder_list,dtype=np.int32)
        result_title=np.array(title_list,dtype=np.int32)
        result_content_len_list=np.array(content_len_list)
        result_title_len_list=np.array(title_len_list)
        result_loss_weight=np.array(loss_weights)

        # shuffle
        result_content_input, result_content_decoder, result_content_len_list,result_title, result_title_len_list, result_loss_weight=\
            self.shuffle(result_content_input,result_content_decoder,result_content_len_list,result_title,result_title_len_list,result_loss_weight)


        num_iter=int(file_size/self.batch_size)
        if self.index<num_iter:
            return_content_input=result_content_input[self.index*self.batch_size:(self.index+1)*self.batch_size]
            return_content_decoder=result_content_decoder[self.index*self.batch_size:(self.index+1)*self.batch_size]

            return_title=result_title[self.index*self.batch_size:(self.index+1)*self.batch_size]
            return_content_len=result_content_len_list[self.index*self.batch_size:(self.index+1)*self.batch_size]
            return_title_len=result_title_len_list[self.index*self.batch_size:(self.index+1)*self.batch_size]
            return_loss_weight=result_loss_weight[self.index*self.batch_size:(self.index+1)*self.batch_size]
            self.index+=1
        else:
            self.index=0
            return_content_input=result_content_input[0:self.batch_size]
            return_content_decoder=result_content_decoder[0:self.batch_size]

            return_title=result_title[0:self.batch_size]
            return_content_len = result_content_len_list[0:self.batch_size]
            return_title_len = result_title_len_list[0:self.batch_size]
            return_loss_weight= result_loss_weight[0:self.batch_size]

        return return_content_input,return_content_decoder,return_title,return_content_len,return_title_len,return_loss_weight

    def get_dev(self,file_name):
        '''
        读取验证数据集
        :return: 
        '''
        train_file = open(PATH+file_name, 'r')

        content_input_list = []
        content_decoder_list = []
        title_list = []
        content_len_list = []
        title_len_list = []
        loss_weights = []

        train_sentcens = train_file.readlines()
        file_size = len(train_sentcens)
        max_index=2000
        for index,sentence in enumerate(train_sentcens):
            if index<=max_index:
                sentence = sentence.replace("\n", "")
                sentences = sentence.split("\t\t")

                content_sentence = sentences[1]
                title_sentence = sentences[0]

                content_array_input, content_array_decoder, _ = self.sent2vec_content(content_sentence, self.content_len)
                title_array, _ = self.sent2vec_title(title_sentence, self.title_len)

                loss_weight = np.zeros_like(content_array_decoder, dtype=np.float32)
                content_len = len(content_sentence.split(" "))
                loss_weight[:content_len] = 1.0
                loss_weights.append(loss_weight)
                content_input_list.append(list(content_array_input))
                content_decoder_list.append(list(content_array_decoder))
                title_list.append(list(title_array))
                if len(str(content_sentence).split(" ")) >= self.content_len:
                    content_len_list.append(self.content_len)
                else:
                    content_len_list.append(len(str(content_sentence).split(" ")))

                if len(str(title_sentence).split(" ")) >= self.title_len:
                    title_len_list.append(self.title_len)
                else:
                    title_len_list.append(len(str(title_sentence).split(" ")))

        train_file.close()
        result_content_input = np.array(content_input_list, dtype=np.int32)
        result_content_decoder = np.array(content_decoder_list, dtype=np.int32)
        result_title = np.array(title_list, dtype=np.int32)
        result_content_len_list = np.array(content_len_list)
        result_title_len_list = np.array(title_len_list)
        result_loss_weight = np.array(loss_weights)
        return result_content_input,result_content_decoder,result_title,result_content_len_list,result_title_len_list,result_loss_weight


    def get_sent(self,sent):

        '''

        :param sent:
        :return:
        '''

        content_input_list = []
        content_decoder_list = []
        title_list = []
        content_len_list = []
        title_len_list = []
        loss_weights = []

        sents_title=' '.join([e for e in jieba.cut(sent)])
        sents_content=" ".join(["BEG"]*self.content_len)



        content_sentence = sents_title
        title_sentence = sents_content

        content_array_input, content_array_decoder, _ = self.sent2vec_content(content_sentence,
                                                                              self.content_len)
        title_array, _ = self.sent2vec_title(title_sentence, self.title_len)

        loss_weight = np.zeros_like(content_array_decoder, dtype=np.float32)
        content_len = len(content_sentence.split(" "))
        loss_weight[:content_len] = 1.0
        loss_weights.append(loss_weight)
        content_input_list.append(list(content_array_input))
        content_decoder_list.append(list(content_array_decoder))
        title_list.append(list(title_array))
        if len(str(content_sentence).split(" ")) >= self.content_len:
            content_len_list.append(self.content_len)
        else:
            content_len_list.append(len(str(content_sentence).split(" ")))

        if len(str(title_sentence).split(" ")) >= self.title_len:
            title_len_list.append(self.title_len)
        else:
            title_len_list.append(len(str(title_sentence).split(" ")))

        result_content_input = np.array(content_input_list, dtype=np.int32)
        result_content_decoder = np.array(content_decoder_list, dtype=np.int32)
        result_title = np.array(title_list, dtype=np.int32)
        result_content_len_list = np.array(content_len_list)
        result_title_len_list = np.array(title_len_list)
        result_loss_weight = np.array(loss_weights)
        return result_content_input, result_content_decoder, result_title, result_content_len_list, result_title_len_list, result_loss_weight

    def  get_Q_array(self,Q_sentence):
        '''
        根据输入问句构建Q矩阵
        :param Q_sentence: 
        :return: 
        '''
        content_len=len(str(Q_sentence).replace("\n","").split(" "))
        if content_len>=self.content_len:
            content_len=self.content_len
        Q_array,_=self.sent2array(Q_sentence,self.content_len)
        return Q_array,np.array([content_len])

    def get_A_array(self,A_sentence):
        '''
        根据输入的答案句子构建A矩阵
        :param A_sentence: 
        :return: 
        '''
        A_sentence, label = self.get_ev_ans(A_sentence)
        title_len=len(label)
        if title_len>=self.title_len:
            title_len=self.title_len
        return self.sent2array(A_sentence,self.title_len)[0],np.array([title_len])

    def get_vocab_size(self):
        '''
        获取词典大小
        :return: 
        '''
        return len(self.content_vocab),len(self.title_vocab)
if __name__ == '__main__':

    dd = DataDealSeq(train_path="/baidu_zd_500.txt", test_path="/test.txt",
                            dev_path="/test.txt",
                            dim=100, batch_size=16 ,content_len=50, title_len=15, flag="train_new")



    result_content_input, result_content_decoder, result_title, result_content_len_list, result_title_len_list, result_loss_weight=dd.next_batch()

    print(result_content_decoder)


