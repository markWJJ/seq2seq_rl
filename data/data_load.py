# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
from __future__ import print_function
import tensorflow as tf
import numpy as np
import codecs
import regex

def load_de_vocab():
    vocab = [line.split()[0] for line in codecs.open('preprocessed/sgns.merge.char.txt', 'r', 'utf-8').read().splitlines() if int(line.split()[1])>=hp.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}

    return word2idx, idx2word

def load_en_vocab():
    vocab = [line.split()[0] for line in codecs.open('preprocessed/sgns.merge.char.txt', 'r', 'utf-8').read().splitlines() if int(line.split()[1])>=hp.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word

def create_data_train(source_sents, target_sents): 
    de2idx, idx2de = load_de_vocab()
    en2idx, idx2en = load_en_vocab()
    
    # Index
    x_list, y_list, Sources, Targets = [], [], [], []
    for source_sent, target_sent in zip(source_sents, target_sents):
        if max(len(source_sent), len(target_sent)) < (hp.maxlen-1):
            x = []
            x.append(de2idx.get("<S>", 1))
            for each in source_sent:
                
                x.append(de2idx.get(each, 1))
            
            x.append(de2idx.get("</S>", 1))
            x_list.append(np.array(x))
            Sources.append(source_sent)

            y=[]
            y.append(de2idx.get("<S>", 1))
            for each in target_sent:
                
                y.append(en2idx.get(each, 1))
            y.append(en2idx.get("</S>", 1))
            y_list.append(np.array(y))
            Targets.append(target_sent)

    
    
    
    # Pad      
    X = np.zeros([len(x_list), hp.maxlen], np.int32)
    Y = np.zeros([len(y_list), hp.maxlen], np.int32)
    for i, (x, y) in enumerate(zip(x_list, y_list)):
        X[i] = np.lib.pad(x, [0, hp.maxlen-len(x)], 'constant', constant_values=(0, 0))
        Y[i] = np.lib.pad(y, [0, hp.maxlen-len(y)], 'constant', constant_values=(0, 0)) #?
    #print(X[0:100])
    #print(Sources[0])
    #print(Sources[1])
    #print(Sources[2])
    #print(Sources[:100])
    #print("==============")
    #print(Y[:100])
    #print(Targets[:100])
    return X, Y, Sources, Targets


def create_data_test(source_sents, target_sents): 
    de2idx, idx2de = load_de_vocab()
    en2idx, idx2en = load_en_vocab()
    
    # Index
    x_list, y_list, Sources, Targets = [], [], [], []
    for source_sent, target_sent in zip(source_sents, target_sents):
        if max(len(source_sent), len(target_sent)) < (hp.maxlen-1):
            x = []
            x.append(de2idx.get("<S>", 1))
            for each in source_sent:
                
                x.append(de2idx.get(each, 1))
            
            x.append(de2idx.get("</S>", 1))
            x_list.append(np.array(x))
            Sources.append(source_sent)

            y=[]
            y.append(de2idx.get("<S>", 1))
            for each in target_sent:
                
                y.append(en2idx.get(each, 1))
            y.append(en2idx.get("</S>", 1))
            y_list.append(np.array(y))
            Targets.append(target_sent)

    
    
    
    # Pad      
    X = np.zeros([len(x_list), hp.maxlen], np.int32)
    Y = np.zeros([len(y_list), hp.maxlen], np.int32)
    for i, (x, y) in enumerate(zip(x_list, y_list)):
        X[i] = np.lib.pad(x, [0, hp.maxlen-len(x)], 'constant', constant_values=(0, 0))
        Y[i] = np.lib.pad(y, [0, hp.maxlen-len(y)], 'constant', constant_values=(0, 0)) #?
    #print(X[0:100])
    #print(Sources[0])
    #print(Sources[1])
    #print(Sources[2])
    #print(Sources[:100])
    #print("==============")
    #print(Y[:100])
    #print(Targets[:100])
    return X, Y, Sources, Targets


def load_train_data():
    syn = []
    t="\\"
    pairs = [regex.sub("[^a-zA-Z0-9\u4e00-\u9fa5\s]", "", pair) for pair in codecs.open(hp.train, 'r', 'utf-8').read().split("\n\n") if t not in pair and len(pair)>=2]
    print("all of train sents %d"%len(pairs))
    co_sents = []
    cx_sents = []
    for i in range(len(pairs)):
        sents = [sent for sent in pairs[i].splitlines()]
        print(sents)
        if len(sents)>=2:
            co_sents.append(sents[0])
            cx_sents.append(sents[1])
    X, Y, Sources, Targets = create_data_train(co_sents, cx_sents)
    print(Sources)
    print("X: %d" % len(X))
    print("Y: %d" % len(Y))
    print("Sources: %d" % len(Sources))
    print("Targets: %d" % len(Targets))
    

    return X, Y
    
def load_test_data():
    syn = []
    def _refine(line):
        line = regex.sub("<[^>]+>", "", line)
        line = regex.sub("[^a-zA-Z0-9\u4e00-\u9fa5\s']", "", line) 
        return line.strip()

    sents = [_refine(line) for line in codecs.open(hp.test, 'r', 'utf-8').read().split("\n")]
    
    o_sents = []
    x_sents = []
    for each in sents:
        if len(each.split()) == 3:
            o_sents.append(each.split()[0])
            x_sents.append(each.split()[-2])

    X, Y, Sources, Targets = create_data_test(o_sents, x_sents)
    return X, Sources, Targets # (1064, 150)

def get_batch_data():
    # Load data
    X, Y = load_train_data()
    print("FINAL DATA")
    print(len(X))
    print(len(Y))
    
    # calc total batch count
    num_batch = len(X) // hp.batch_size #double slashes mean integer division
    
    # Convert to tensor
    X = tf.convert_to_tensor(X, tf.int32)
    Y = tf.convert_to_tensor(Y, tf.int32)
    
    # Create Queues
    input_queues = tf.train.slice_input_producer([X, Y])
            
    # create batch queues
    x, y = tf.train.shuffle_batch(input_queues,
                                num_threads=8,
                                batch_size=hp.batch_size, 
                                capacity=hp.batch_size*64,   
                                min_after_dequeue=hp.batch_size*32, 
                                allow_smaller_final_batch=False)
    
    return x, y, num_batch # (N, T), (N, T), ()

