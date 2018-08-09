# import numpy as np
# import tensorflow as tf
#
#
#
# s=tf.placeholder(shape=(3,4,5),dtype=tf.int32)
# batch=tf.placeholder(shape=(5,),dtype=tf.int32)
# index=tf.placeholder(shape=(1,4,1),dtype=tf.int32)
# # ss=s[batch,index]
# ss=tf.multiply(s,index)
# ss=tf.reduce_max(ss,1)
#
# with tf.Session() as sess:
#     s1 = np.ones(shape=(3,4,5))
#
#     index1 = np.array([0, 1, 0,0])
#     index1=np.reshape(index1,(1,4,1))
#     batch1=np.arange(5)
#     ss1=sess.run(ss,feed_dict={s:s1,
#                            batch:batch1,
#                            index:index1})
#
#     print(ss1)
# s1 = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
#

# from nltk import bleu
# true_dec=['1','2','3']
# this_dec=['4','3','4']
#
# bleu_score1=bleu([true_dec],this_dec)
#
# print(bleu_score1)

# import numpy as np
#
# s=np.ones(shape=(3,4,5))
#
# ss=s[[0,1,2],[0,1,3],[]]
#
# print(ss)
# import numpy as np
# x=[1,2,3]
# y=[2,2,2]
# s=sum(1 for e in np.equal(x,y) if e)
# print(s)

# import random
# s = random.uniform(0, 1)
# print(s)
import numpy as np
from numpy.random import random
from numpy.random import randn
from numpy.random import randint
s=np.ones(shape=(2,3,4))

ss=randint(0,800,size=(15))
print(ss)
