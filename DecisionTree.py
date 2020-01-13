#Ruchit Desai - RBD2127
#Emily Hao -  ESH2160

import numpy as np
import pandas as pd
import math


class Node():
    def __init__(self, root, left, right):
        self.root           = root
        self.left           = left
        self.right          = right

    def set_root(self, root):
        self.root           = root

    def set_right(self, right):
        self.right          = right

    def set_left(self, left):
        self.left           = left

class DecisionTree():
    V_mag               = int()
    test                = list()
    ham_sum             = list()
    spam_sum            = list()
    y_pred              = list()
    k                   = 0
    p                   = 0
    vocab               = dict()

    def __init__(self, train, test, vocab):
        #load variables on init
        self.V_mag          = len(vocab)
        self.vocab          = vocab
        self.test           = test
        self.train(train)
        print("***TRAINING COMPLETE -- Decision Tree Classifier***")

    def id3(self, examples, target_attribute, attributes):

            pass

    def split(self, index, ham_sum, spam_sum, magnitude):
        pass

    def createTree(self, ham, spam):
        root_node           = Node(None, None, None)

        ex_ham              = np.sum([x for x in ham])
        ex_spam             = np.sum([x for x in spam])

        if ex_ham == 0 and ex_spam != 0: 
            root_node.set_root(['Spam'])
            return root_node
        
        if ex_spam == 0 and ex_ham != 0: 
            root_node.set_root('Ham')
            return root_node

        if len(ham) == 1 and len(spam) == 1:
            if ex_ham > ex_spam:
                root_node.set_root('Ham')
            else:
                root_node.set_root('Spam')


        ham_sum                 = np.sum(ham, axis=0)
        spam_sum                = np.sum(spam, axis=0)
        total_words             = len(ham)
        gain_vocab              = list()

        entropy_of_h_s           = self.entropy(len(ham),len(spam), len(ham)+len(spam))

        for ham_freq , spam_freq in zip(self.ham_sum, self.spam_sum):
            #print("Ham: " + str(ham_freq) + " . Spam: " + str(spam_freq))
            x = self.entropy(ham_freq, spam_freq, self.V_mag)
            gain = entropy_of_h_s - x
            gain_vocab.append(gain)

        index_min_entropy       = gain_vocab.index(min(x for x in gain_vocab if x!=0.0))

        sp_l_ham , sp_l_spam, sp_r_ham , sp_r_spam , voc_l, voc_r= self.split(index_min_entropy, ham_sum, spam_sum, voc)
        
        root_node.set_left      = createTree(sp_l_ham, sp_l_spam)
        root_node.set_right     = createTree(sp_r_ham, sp_r_spam)

        return root_node
        voc = list(self.vocab.items())

        #self.id3(train, , self.vocab)
        print("Length of gain_vocab: " + str(len(gain_vocab)))

        print("max gain word: " + str(voc[index_min_entropy]))


        #print("Entropy Total: " + str(np.sum(gain_vocab,axis=0)))

    def train(self, train):
        #train = n x m where train[n][m] = train[all words and their frequencies in current document][true label of current document]
        print("Length of Train: " + str(len(train)) + "Length of Test: " + str(len(self.test)))
        ham            = list()
        spam           = list()
        
        #split train set into ham / spam
        for data in train:
            if data[1] == 1.0:
                ham.append(data[0])
            elif data[1] == 0.0:
                spam.append(data[0])

        tree                     = createTree(ham, spam)
        
    def entropy(self, x, y, n):
        #print("x: " + str(x) +" y: " + str(y) +" n: " + str(n))
        if x != 0 and y !=0 : 
            return -((x/n) * np.log2(x/n)) - ((y/n) * np.log2(y/n))

        return 0.0

    def predict(self):
        # INCOMPLETE ! ! ! ! ! DO NOT USE.
        return(np.ones(len(self.test)))
        '''
        if classify > 0 and test_doc[1] == 1.0:
            prediction = 1
        else:
            prediction = 0

        self.y_pred.append(prediction)

        return self.y_pred
        '''