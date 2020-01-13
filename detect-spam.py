# tmatrixhy

import math
import os.path
import sys
import glob
import errno
import pickle

#data-science classes
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt

#custom classes
import porter2 as clean
from NaiveBayes import Naive_Bayes_Classifier as NB
from KNN import K_Nearest_Neighbors as KNN
from DecisionTree import DecisionTree as DT


'''
Created a construct for the dataset, however I found sklearn's tutorials and the CountVectorizer was
perfect for what we were trying to achieve, hence this is commented out.

class bag_of_words:
    bag         = dict()

    def __init__(self, all_doc_lines):
        unsorted_words = []
        self.bag = dict()
        for line in all_doc_lines:
            #print(str(line))
            v_line = str(line).split()
            for word in v_line:
                clean_word = clean.stem(word)        #clean up word
                #print("WORD: " + str(word) + " CLEAN WORD :: " + str(clean_word))
                #if clean_word not in unsorted_words:
                if clean_word.isalpha():
                    unsorted_words.append(clean_word)


        #sort words here
        sorted_words = sorted(unsorted_words, key=str.lower)
        #create hash value for word based on ascii + base amt
        for word in sorted_words:
            if word not in self.bag.keys():
                self.bag[word] = 1
            else:
                val = self.bag.get(word)
                val+=1
                self.bag[word] = val
                #print("WORD: " + str(clean_word) + " ALREADY IN BAG AS :: " + str(hash_of_word))
        #print(self.bag)
        #if (word != clean_word):
            #print("Converted " + word + " to " + clean_word)
        
        #print("Hash of: " + str(clean_word) + " ---> " + str(hash_of_word))

    def hash_this(self, word):
        key         = 1337
        counter     = 1
        for x in word:
            key+=ord(x)*counter
            counter += 1

        return key

    def get_frequency(self, word):
        return self.bag.get(word)

    def get_length(self):
        return len(self.bag)

'''

def load_files(path):
    #open file code from https://askubuntu.com/questions/352198/reading-all-files-from-a-directory   
    files = glob.glob(path)   
    x = list()
    print("Reading files from : " + str(path))
    for name in files: # 'file' is a builtin type, 'name' is a less-ambiguous variable name.
        try:
            with open(name, encoding="utf8", errors='ignore') as f: 
                all_words = []
                new_file = f.read()
                all_lines_tok = word_tokenize(new_file)
                for word in all_lines_tok:
                    clean_word = clean.stem(word)
                    if clean_word.isalpha():
                        all_words.append(clean_word.lower())
                sorted(all_words)
                x.append(' '.join(all_words))

        except IOError as exc:
            if exc.errno != errno.EISDIR: # Do not fail if a directory is found, just ignore it.
                raise

    print("Files Loaded: " + str(len(x)))

    return x

def eval_metrics(Yi_test, prediction):
    #confusion matrix ref: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    
    # tn = true negative
    # fp = false positive
    # fn = false negative
    # tp = true positive
    # error rate alternate is ( 1 - accuracy )

    tn, fp, fn, tp      = confusion_matrix(Yi_test, prediction).ravel()
    print("-----<Evaluation Metrics>-----")
    print("True Positive : " + str(tp))
    print("False Positive: " + str(fp))
    print("True Negative : " + str(tn))
    print("False Negative: " + str(fn))
    print("------------------------------")
        
    accuracy            = (tp+tn)/(tp+tn+fp+fn)
    precision           = tp / (tp+fp)
    sensitivity         = tp / (tp+fn)
    specificity         = tn / (tn+fp)
    error_rate          = (fp + fn) / len(Yi_test)

    return [accuracy , precision, sensitivity, specificity, error_rate]

def cross_validation(data, vocab, k):
    #Kfold Reference from https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    eval_all_folds      = []
    KFld                = KFold(n_splits=k, shuffle=True, random_state=None)
    fold                = 0
    class_names         = [0,1]
    print(">>>>RUNNING CROSS VALIDATION: k = " + str(k))
    for train_index, test_index in KFld.split(data[0],data[1]):
        fold                += 1
        Xi_train            = data[0][train_index]
        Xi_test             = data[0][test_index]
        Yi_train            = data[1][train_index]
        Yi_test             = data[1][test_index]

        train_data          = list(zip(Xi_train, Yi_train))
        test_data           = list(zip(Xi_test, Yi_test))
        
        #naive bayes metrics
        naive_bayes         = NB(train_data,test_data, vocab)
        nb_metrics          = eval_metrics(Yi_test, naive_bayes.predict())

        #knn metrics p=inf, p=1, p=2 ; k = 100

        knn                 = KNN(train_data, test_data, vocab, 5)
        knn_metrics_inf     = eval_metrics(Yi_test, knn.predict('inf'))

        knn_metrics_p_1     = eval_metrics(Yi_test, knn.predict(1))
        
        knn_metrics_p_2     = eval_metrics(Yi_test, knn.predict(2))

        #dt metrics
        '''
        dt                  = DT(train_data, test_data, vocab)
        dt_metrics          = eval_metrics(Yi_test, dt.predict())
        '''
        eval_all_folds.append([nb_metrics, knn_metrics_inf, knn_metrics_p_1, knn_metrics_p_2])#, dt_metrics])

        print("FOLD # " + str(fold) + ": " + str([nb_metrics, knn_metrics_inf, knn_metrics_p_1, knn_metrics_p_2]))#, dt_metrics]))
    
    return np.mean(eval_all_folds, axis=0)

def output_plot(k, nb, knn_inf, knn_1, knn_2):#, dt):
    plots = ['Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'Error Rate']

    nb_T            = np.transpose(nb)
    knn_inf_T       = np.transpose(knn_inf)
    knn_1_T         = np.transpose(knn_1)
    knn_2_T         = np.transpose(knn_2)
    #dt_T            = np.transpose(dt)
    counter = 1

    for plot in plots:
        #plt.figure(counter)
        plt.subplot(3,2,counter)
        plt.plot(k, nb_T[counter-1], label='Naive Bayes', linestyle='--', marker = 'o', color='b')
        plt.plot(k, knn_inf_T[counter-1], label='K-NN; p = inf', linestyle='-.', marker = 'v', color='darkred')
        plt.plot(k, knn_1_T[counter-1], label='K-NN; p = 1', linestyle='-.', marker = 'v', color='indianred')
        plt.plot(k, knn_2_T[counter-1], label='K-NN; p = 2', linestyle='-.', marker = 'v', color='lightcoral')
        #plt.plot(k, dt_T[counter-1], label='Decision Tress', linestyle='-', marker = 's', color='g')

        plt.title(plot)
        plt.ylabel(str(plot) + str(" %"))

        counter+=1
    print("Naive Bayes:")
    print(nb_T)
    print("KNN - p = inf:")
    print(knn_inf_T)
    print("KNN - p = 1  :")
    print(knn_1_T)
    print("KNN - p = 2  :")
    print(knn_2_T)
    plt.figlegend(['Naive Bayes', 'K-NN; p = inf', 'K-NN; p = 1', 'K-NN; p = 2'], loc =4, ncol=1)
    plt.savefig('metrics.png')
    plt.show()

def main():
    if len(sys.argv) > 1:
        ham                 = sys.argv[1]
        spam                = sys.argv[2]

        D_ham               = load_files(ham)
        D_spam              = load_files(spam)
    
        filesave = open("D_ham.obj","wb")
        pickle.dump(D_ham,filesave)
        filesave.close()
        filesave = open("D_spam.obj","wb")
        pickle.dump(D_spam,filesave)
        filesave.close()
    else:
        if os.path.isfile('D_ham.obj') and os.path.isfile('d_spam.obj'):
            file = open("D_ham.obj",'rb')
            D_ham = pickle.load(file)
            file.close()
            file = open("D_spam.obj",'rb')
            D_spam = pickle.load(file)
            file.close()
        else:
            print("Usage: python hw1.py <full path of ham directory> <full path of spam directory>")
            print("   ex: python hw1.py c:\\enron1\\ham\\* c:\\enron1\\spam\\*")

    #generate n x m array of documents x (words in vocabulary)
    #where array[n][m] = frequency of word m in given document n
    combined            = D_ham + D_spam
    count_vect          = CountVectorizer()
    dataset_vec         = count_vect.fit_transform(combined)
    doc_vector          = dataset_vec.toarray()

    #establish vocabulary (not really needed but here for experiments)
    features            = count_vect.get_feature_names()
    frequency           = np.asarray(doc_vector.sum(axis=0))
    vocab               = dict(zip(features,frequency))

    #generate full dataset with classifiers
    full_dataset        = [np.asarray(doc_vector),np.concatenate((np.ones(len(D_ham)),np.zeros(len(D_spam))))]
    
    avg_CM_NB           = []
    avg_CM_KNN_inf      = []
    avg_CM_KNN_p_1      = []
    avg_CM_KNN_p_2      = []
    #avg_CM_DT           = []

    #run cross validation for #folds(k) = 2, 5, 8, 11
    k_folds = [2,5,8,10]
    for k in k_folds:
        nb, knn_inf, knn_p_1, knn_p_2 = cross_validation(full_dataset, vocab, k)
        avg_CM_NB.append(nb)
        avg_CM_KNN_inf.append(knn_inf)
        avg_CM_KNN_p_1.append(knn_p_1)
        avg_CM_KNN_p_2.append(knn_p_2)
        #avg_CM_DT.append(dt)
    
    output_plot(k_folds, avg_CM_NB, avg_CM_KNN_inf, avg_CM_KNN_p_1, avg_CM_KNN_p_2) #, avg_CM_DT)

if __name__ == "__main__":
    main()