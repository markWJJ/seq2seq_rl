
import regex
import codecs

def load_train_data():
    syn = []
    t="\\"
    pairs = [regex.sub("[^a-zA-Z0-9\u4e00-\u9fa5\s]", "", pair) for pair in codecs.open('./zhidao_train.txt', 'r', 'utf-8').read().split("\n\n") if t not in pair and len(pair)>=2]
    print("all of train sents %d"%len ( pairs))
    co_sents = []
    cx_sents = []
    for i in range(len(pairs)):
        sents = [sent for sent in pairs[i].splitlines()]
        print(sents)
        if len(sents)>=2:
            co_sents.append(sents[0])
            cx_sents.append(sents[1])
    # X, Y, Sources, Targets = create_data_train(co_sents, cx_sents)
    # print(Sources)
    # print("X: %d" % len(X))
    # print("Y: %d" % len(Y))
    # print("Sources: %d" % len(Sources))
    # print("Targets: %d" % len(Targets))



if __name__ == '__main__':
    load_train_data()