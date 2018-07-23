#!/usr/bin/python
#-*- encoding:utf-8 -*-
import jieba
import re
jieba.load_userdict('./user_dict.txt')

class data_deal(object):

    def __init__(self):
        pass

    def __data_deal__(self,filen_name):

        data_dict={}
        this_id=None
        ss=[]
        for ele in open(filen_name,'r').readlines():
            ele=ele.replace('\n','')
            try:
                sent=ele.split('\t\t')[1]
                qa_id=ele.split('\t\t')[0].split('\t')[0]
                user_type=ele.split('\t\t')[0].split('\t')[2]
            except:
                print(ele)

            if this_id != qa_id and ss:
                if qa_id not  in data_dict:
                    data_dict[this_id]=ss
                else:
                    s_=data_dict[this_id]
                    s_.extend(ss)
                    data_dict[this_id]=s_
                this_id = qa_id
                ss=[(user_type,sent)]
            else:
                this_id=qa_id
                ss.append((user_type,sent))

        new_dict={}
        for k,v in data_dict.items():
            ss=[]
            this_label=v[0][0]
            this_ele=''
            for i in range(len(v)):
                if this_label==v[i][0]:
                    this_ele+=v[i][1]+','
                elif this_label!=v[i][0]:
                    ss.append((this_label,this_ele))
                    this_label=v[i][0]
                    this_ele=v[i][1]
                    if i==len(v)-1:
                        ss.append((this_label, this_ele))
                elif i==len(v)-1:
                    ss.append((this_label,this_ele))
            new_dict[k]=ss

        return new_dict


    def qa_generate(self,data_dict):

        qa_dict={}
        for k,v in data_dict.items():
            qa_dict[k]=[]
            print(v)
            for i in range(len(v)-1):
                if v[i][0]=='0':
                    qa_dict[k].append({'question':v[i][1],'answer':v[i+1][1]})
                    i+=2


        return qa_dict

    def sent_deal(self,sent):
        sub_pattern='[ORDERID_\d.]'
        sent=re.subn(sub_pattern,'[订单号x]',sent)[0]
        sent=re.subn('\\[数字x\\]','数字x',sent)[0]
        sent=re.subn('\\[金额x\\]','金额x',sent)[0]
        sent=re.subn('\\[姓名x\\]','姓名x',sent)[0]
        sent=re.subn('\\[日期x\\]','日期x',sent)[0]
        sent=re.subn('\\[订单号x\\]','订单号x',sent)[0]
        sent=re.subn('#E-s\\[数字x\\]','E-s数字x',sent)[0]
        sent=re.subn('\\[订单号x\\]','订单号x',sent)[0]

        return sent

    def build_train_data(self,qa_dict,data_name):
        fw=open(data_name,'w')
        for k, v in qa_dict.items():
            for ele in v:
                ques=' '.join([e for e in jieba.cut(self.sent_deal(ele['question']))])
                fw.write(ques)
                fw.write('\t\t')
                ans=' '.join([e for e in jieba.cut(self.sent_deal(ele['answer']))])
                fw.write(ans)
                fw.write('\n')

if __name__ == '__main__':

    dd=data_deal()
    data_dict=dd.__data_deal__('./jd_data.txt')
    qa_dict=dd.qa_generate(data_dict)

    # for k,v in qa_dict.items():
    #     print(k,v)
    #     print('\n')
    dd.build_train_data(qa_dict,'write.txt')
