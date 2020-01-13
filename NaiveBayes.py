# tmatrixhy

import numpy as np
import pandas as pd 

class Naive_Bayes_Classifier():
    P_ham               = float()
    P_spam              = float()
    V_mag               = int()
    ham                 = list()
    spam                = list()
    ham_conditionals    = list()
    spam_conditionals   = list()
    y_pred              = list()
    test                = list()


    def __init__(self, train, test, vocab):
        #load variables on init
        self.V_mag          = len(vocab)
        self.test           = test
        self.train(train)
        print("***TRAINING COMPLETE -- Naive Bayes Classifier***")

    def train(self, train):
        self.ham            = list()
        self.spam           = list()
        
        #split train set into ham / spam
        for data in train:
            if data[1] == 1.0:
                self.ham.append(data[0])
            elif data[1] == 0.0:
                self.spam.append(data[0])

        #priors
        self.P_ham          = len(self.ham) / len(train)
        self.P_spam         = len(self.spam) / len(train)
        
        #conditionals
        #P(word| class) = (# times word shows up in class + 1(lapace)) / (#words in class + #words in vocab)

        #array for # of times the word shows up in the class
        ham_sum             = np.sum(self.ham, axis=0)
        spam_sum            = np.sum(self.spam, axis=0)

        #sum of words in class (above array)
        beta_ham            = np.sum(ham_sum, axis=0)
        beta_spam           = np.sum(spam_sum, axis=0)
        
        self.ham_conditionals    = list()
        self.spam_conditionals   = list()

        #generate conditionals with Lapace smoothing
        for x in ham_sum:
            conditional     = (x + 1) / (beta_ham + self.V_mag)
            self.ham_conditionals.append(conditional)

        for x in spam_sum:
            conditional     = (x + 1) / (beta_spam + self.V_mag)
            self.spam_conditionals.append(conditional)

        #print("ham conditionals:       " + str(self.ham_conditionals) + "  . length     : " + str(len(self.ham_conditionals)))
        #print("spam conditionals:      " + str(self.spam_conditionals) + "  . length     : " + str(len(self.spam_conditionals)))

    def predict(self):
        #predict using highest un-normalized log probability score.
        h_correct                     = 0
        h_incorrect                   = 0
        s_correct                     = 0
        s_incorrect                   = 0
        self.y_pred                   = list()

        for Di in self.test:
            #Di[1] == y_hat == Actual class of given test example
            #len(Di[0]) == 38038 fixed (vocab length)

            #get sum of log(conditionals)
            ham_conditional_total   = 0.0
            spam_conditional_total  = 0.0
            prediction              = int()

            for word_count , con_ham , con_spam in zip(Di[0], self.ham_conditionals, self.spam_conditionals):
                if word_count != 0:
                    #print("Word Count : " + str(word_count) + ".    con_ham: " + str(con_ham) + ".     con_spm: " + str(con_spam) )
                    ham_conditional_total  += (np.log(con_ham) * word_count)
                    spam_conditional_total += (np.log(con_spam) * word_count)
            
            #classify
            ham_class               = np.log(self.P_ham) + ham_conditional_total
            spam_class              = np.log(self.P_spam) + spam_conditional_total
            #print("ham: " + str(ham_class) + "     spam: " + str(spam_class) + "                    " + str(self.P_ham) + "   " + str(self.P_spam))


            #one lingering question: should it be > than or >= ?
            if ham_class >= spam_class:
                prediction = 1
            else:
                prediction = 0

            self.y_pred.append(prediction)

        return self.y_pred