# tmatrixhy

import numpy as np
import pandas as pd

class K_Nearest_Neighbors():
    V_mag               = int()
    test                = list()
    ham_sum             = list()
    spam_sum            = list()
    y_pred              = list()
    k                   = 0
    p                   = 0

    def __init__(self, train, test, vocab, k):
        #load variables on init
        self.k              = k
        self.V_mag          = len(vocab)
        self.test           = test
        self.train(train)
        print("***TRAINING COMPLETE -- KNN Classifier***")

    def p_norm_dist(self, Xi, Xj):
        if self.p == 'inf':
            return max(Xi, Xj)

        total = (Xj-Xi)**self.p

        return total**(1/self.p)

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

        self.ham_sum             = np.sum(ham, axis=0)
        self.spam_sum            = np.sum(spam, axis=0)

    def predict(self, p):
        self.p = p
        counter = 0
        #test = reversed(self.test)
        self.y_pred = list()
        for test_doc in self.test:
            NN      = list()
            for a, b, c in zip (self.spam_sum, test_doc[0], self.ham_sum):
                if b != 0:
                    ham_dist = abs(self.p_norm_dist(a, b))
                    spam_dist = abs(self.p_norm_dist(c, b))
                    #print("Ham dist: " + str(ham_dist) + ". Spam dist: " + str(spam_dist))
                    #if ham_dist > 1 and spam_dist > 1:
                    if ham_dist < spam_dist:
                        NN.append((ham_dist, 1))
                    else:
                        NN.append((spam_dist, -1))
                #REMOVE THIS FOR FINAL RUN---------------------------------------------------------------------------<<<< READ >>>> ---------
                '''
                if len(NN) > self.k:
                    break
                '''
            #sort so that the lowest distances are first
            sorted_NN = sorted(NN, key=lambda x: x[0])

            #take majority voting of self.k nearest neighbors
            classify = sum(k for j,k in NN[0:self.k])

            #print("Doc " + str(counter) + " is classified: " + str(classify) + ". " + "Actual: " + str(test_doc[1]))

            prediction = int()
            if classify > 0 and test_doc[1] == 1.0:
                prediction = 1
            else:
                prediction = 0

            self.y_pred.append(prediction)

        return self.y_pred